# Decoder preliminar FP32 de características gaussianas

[English](../docs/gaussian-decoder.md)

`idol_mlx.gaussian_decoder.GaussianFixedlenDecoder` implementa la pasada FP32 de
características del [decoder gaussiano original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
Recibe anclas `[B, L, 3]` y condiciones `[B, Lc, C]`, y devuelve características
sin procesar `[B, L, num_gaussians * 15]`. La conversión a posición, escala,
rotación, opacidad y color es trabajo separado del issue #31; estas
características todavía no son una representación 3D renderizable.

`idol_mlx.gaussian_weights.load_gaussian_checkpoint` valida estrictamente el
esquema completo FP32 con nombres originales antes de modificar el modelo,
incluidos los buffers `base_offset_scale` escalar y `points_offset_perturbation`.
No convierte ni ignora pesos. La API no descarga checkpoints. El modelo admite
posiciones 3D e inferencia FP32 explícita; la precisión mixta original no está
validada. Las comparaciones con pesos y entradas sintéticas pasan en CPU y Metal.

## Comparación reducida con checkpoint entrenado

Se cargó estrictamente `decoder_gs.safetensors` (328 tensores FP32) de
[`speridlabs/idol` revisión `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902)
y se comparó con PyTorch ejecutado explícitamente en FP32, en un Apple M4 con
32 GiB, macOS 27.2, Python 3.13.14, MLX 0.32.3 y PyTorch 2.15.0. Cada ejecución
usó lote 1 y entradas sintéticas FP32. La tolerancia provisional fue
`atol=2e-5`, `rtol=2e-4`; el error relativo influye en el resultado.

| Dispositivo MLX | Anclas | Tokens de condición | Semilla | Error absoluto máximo de características | Valores que fallan | Resultado |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | 8 | 4 | 2026 | 1.33514e-5 | 0 / 3840 | Pasa |
| GPU Metal | 8 | 4 | 2026 | 2.47955e-5 | 0 / 3840 | Pasa |
| CPU | 32 | 8 | 42 | 1.64509e-5 | 0 / 15360 | Pasa |
| GPU Metal | 32 | 8 | 42 | 5.25117e-5 | 6 / 15360 | **Falla** |

Los embeddings de posición pasaron en los cuatro casos. Establecer
`MLX_ENABLE_TF32=0` no cambió el fallo observado en GPU con 32 anclas. La causa
de la diferencia restante no está determinada; el issue #35 continúa abierto
y no se ampliaron las tolerancias. Estas entradas reducidas no validan conteos
completos de anclas, conversión de atributos, precisión mixta ni inferencia
integral. No se publicaron pesos ni entradas.

## Reproducir

Con el SDK original fijado en `PYTHONPATH`, usa configuración y pesos locales
confiables obtenidos bajo sus condiciones aplicables. El diagnóstico reserva los
modelos completos de referencia y MLX e imprime solo métricas agregadas. El
código 0 indica que ambas comparaciones pasaron y el 1 que alguna falló.

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_gaussian_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder_gs.safetensors \
  --anchors 32 --context-tokens 8 --seed 42 --device gpu
```

Usa `--device cpu` para MLX CPU. CI no necesita el checkpoint entrenado.
