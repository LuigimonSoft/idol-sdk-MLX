# Conversión de atributos gaussianos

[English](../docs/gaussian-attributes.md)

`idol_mlx.gaussian_attributes.decode_gaussian_attributes` convierte anclas y
características FP32 del decoder gaussiano preliminar en arrays MLX por lote
según las [convenciones originales fijadas](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
Proporciona la configuración de representación y los buffers
`points_offset_perturbation` y `base_offset_scale` del decoder cargado.

```python
from idol_mlx.gaussian_attributes import decode_gaussian_attributes

# model es un GaussianFixedlenDecoder cargado; args son sus argumentos de configuración.
features = model(anchors, condition)
attributes = decode_gaussian_attributes(
    anchors, features, representation_config=args["representation_config"],
    points_offset_perturbation=model.points_offset_perturbation,
    base_offset_scale=model.base_offset_scale,
)
```

Para lote `B`, anclas `L` y `K` gaussianas por ancla, el resultado contiene:

| Clave | Forma | Convención |
| --- | --- | --- |
| `xyz` | `[B, L*K, 3]` | posiciones con eje z hacia arriba, con el desplazamiento original y resta de 0.5 |
| `features_dc` | `[B, L*K, 1, 3]` | coeficientes de armónicos esféricos de grado cero, no RGB |
| `scaling` | `[B, L*K, 3]` | escala softplus con sesgo y kernel mínimo configurados |
| `rotation` | `[B, L*K, 4]` | cuaternión `(w, x, y, z)` normalizado con desplazamiento identidad |
| `opacity` | `[B, L*K, 1]` | sigmoide con sesgo de opacidad configurado |

La API original devuelve una lista de objetos Gaussian por lote; esta devuelve
arrays por lote con valores equivalentes por elemento. Un cuaternión cero antes
de normalizar permanece en cero, igual que en la referencia. Los desplazamientos
de posición no se recortan y la sigmoide puede saturarse en 0 o 1. Las entradas
deben ser arrays MLX FP32 no vacíos y de formas compatibles; se validan sesgos
y formas de buffers. Esta función no exporta PLY ni rota coordenadas a eje y;
para ello usa el [exportador PLY](gaussian-ply.md) por separado.

Las pruebas sintéticas comparan los cinco atributos con PyTorch en CPU y Metal,
incluidos varios lotes y cantidades por ancla, cuaterniones degenerados, color
SH y saturación. Una prueba de integración sintética también compara atributos
generados a partir de las características de la red gaussiana de cada entorno. Con `decoder_gs.safetensors` de
[`speridlabs/idol` revisión `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902),
32 anclas sintéticas, 8 tokens de condición y semilla 42, todos los atributos
pasaron `atol=2e-5`, `rtol=2e-4` en CPU y Metal. En Metal el error absoluto
máximo fue `5.96046e-8` para posición, `0` para color SH, `7.45058e-8` para
escala, `0` para rotación y `1.19209e-7` para opacidad. Un segundo caso Metal
(64 anclas, 16 tokens, semilla 123) también pasó las cinco comparaciones. Las
mediciones usaron Apple M4, macOS 27.2, Python 3.13.14, MLX 0.32.3 y PyTorch 2.15.0.

**El diagnóstico suministra las mismas características generadas por PyTorch a
ambos convertidores** para aislar la conversión de atributos. No valida las
características generadas por la red MLX, que todavía presenta un fallo de
paridad en Metal documentado en el [diagnóstico del decoder](gaussian-decoder.md),
ni valida exportación PLY, precisión mixta o generación integral. No se
publicaron pesos ni entradas.

Para reproducir con archivos locales confiables y el SDK fijado en `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_gaussian_attributes \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder_gs.safetensors \
  --anchors 32 --context-tokens 8 --seed 42 --device gpu
```

Usa `--device cpu` para MLX CPU. El diagnóstico reserva ambos modelos completos,
imprime métricas agregadas y devuelve 0 solo si pasan las cinco comparaciones.
