# Red preliminar de probabilidades octree

[English](../docs/octree-network.md)

`idol_mlx.octree.OctreeProbabilityFixedlenDecoder` implementa la pasada FP32 del
[decodificador octree original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
Recibe centros de celda `[B, L, 3]`, nivel `[B]` y características de condición
`[B, Lc, cond_channels]`, y devuelve logits `[B, L, 8]`. Aplica
`mx.softmax(logits, axis=-1)` para obtener las probabilidades de los hijos.
El muestreo y recorrido de tamaño variable del octree no se implementan aquí.

`idol_mlx.octree_weights.load_octree_checkpoint` lee un archivo safetensors FP32
local y confiable con los nombres originales. Comprueba el esquema completo de
origen y destino, formas y tipos antes de actualizar el modelo; no convierte
valores, cambia formas ni ignora tensores. Esta API no incluye ni descarga pesos.

Las pruebas sintéticas comparan logits y probabilidades con los mismos pesos de
PyTorch y varios lotes y longitudes, con tolerancias provisionales `atol=2e-5`,
`rtol=2e-4`. La referencia se convierte explícitamente a FP32; no se valida su
comportamiento habitual de precisión mixta. También se verifica que checkpoints
inválidos no modifiquen el modelo. CI ejecuta las pruebas en MLX CPU; también se
probaron localmente con Metal. Esta red preliminar no es un muestreador octree ni
un decodificador integral.

## Diagnóstico reducido con checkpoint entrenado

Se cargó estrictamente `decoder.safetensors` (60 tensores FP32) de
[`speridlabs/idol` revisión `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902)
y se comparó con la referencia ejecutada explícitamente en FP32. En un Apple
M4, MLX 0.32.3, PyTorch 2.15.0 y Python 3.12. Los mismos resultados se
reprodujeron con la versión compatible Python 3.13.14. Con lote 1, 8 celdas sintéticas,
4 tokens de condición, semilla 2026 y nivel 4, la implementación nativa antes
de corregir las frecuencias canónicas **no cumplió** las tolerancias
provisionales `atol=2e-5, rtol=2e-4`:

| Dispositivo MLX | Error máximo de posición | Error máximo de logits (fallos / 64) | Error máximo de probabilidad (fallos / 64) |
| --- | ---: | ---: | ---: |
| CPU | 0.00189202 | 0.00456715 (12) | 0.000149608 (5) |
| GPU Metal | 0.00165553 | 0.00364113 (12) | 0.000131965 (5) |

El embedding de posición difería antes de los bloques del decoder. Al usar las
frecuencias de la referencia, el error posicional disminuyó y la comparación
reducida original pasó. La implementación MLX nativa usa ahora esas frecuencias
FP32 canónicas derivadas de la configuración oficial del decoder. Tras la
corrección, con Python 3.13.14, el mismo checkpoint y las tolerancias
provisionales, las tres métricas pasaron en cada escenario:

| Dispositivo | Celdas | Tokens de condición | Semilla | Nivel | Error máximo de logits | Error máximo de probabilidad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 8 | 4 | 2026 | 4 | 2.38419e-5 | 1.90735e-6 |
| CPU | 16 | 7 | 42 | 2 | 1.71661e-5 | 1.65403e-6 |
| CPU | 32 | 8 | 123 | 8 | 3.09944e-5 | 5.69224e-6 |
| GPU Metal | 8 | 4 | 2026 | 4 | 2.52724e-5 | 3.21865e-6 |
| GPU Metal | 16 | 7 | 42 | 2 | 2.92659e-5 | 2.68221e-6 |
| GPU Metal | 32 | 8 | 123 | 8 | 5.91278e-5 | 7.09295e-6 |

El resultado depende tanto del error absoluto como del relativo. El error máximo
de posición fue 1.19209e-7 en CPU y 1.78814e-7 en Metal en las seis ejecuciones.
Son pasadas FP32 independientes con lote 1 y entradas sintéticas; no validan
precisión mixta, recorrido completo del octree, el decoder gaussiano ni la
inferencia integral. No se incluyen pesos ni entradas en el repositorio.

Para reproducirlo, con el SDK original fijado en `PYTHONPATH` y archivos
locales confiables, ejecuta el comando siguiente; usa `--device cpu` para MLX
CPU. El código 0 indica que pasaron las comparaciones de posición, logits y
probabilidades; el 1 indica que falló alguna. Solo imprime métricas agregadas y
carga todos los pesos de la red.

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_octree_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors --device gpu
```
