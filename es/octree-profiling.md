# Observaciones de límites y memoria del octree

[English](../docs/octree-profiling.md)

Se midió el muestreador octree FP32 preliminar en un Apple M4 con 32 GiB de
memoria unificada, macOS 27.2, Python 3.13.14, MLX 0.32.3 y Metal. Se cargaron
estrictamente `decoder.safetensors` (60 tensores FP32) y la configuración de
[`speridlabs/idol` revisión `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902).
Las entradas fueron sintéticas FP32, lote 1, semilla 101 y 8192 tokens de
condición. Cada fila corresponde a una ejecución en un proceso nuevo; los
tiempos no son un benchmark estadístico.

| Anclas | Niveles | Tiempo (s) | Pico activo MLX (MiB) | Caché MLX final (MiB) | Pico RSS del proceso (MiB) | Resultado |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8 | 0.00012 | 194.3 | <0.1 | 246.3 | Pasa; no ejecuta el modelo |
| 1 | 8 | 0.219 | 708.6 | 514.4 | 253.5 | Pasa |
| 64 | 4 | 0.251 | 733.1 | 544.8 | 254.2 | Pasa |
| 1024 | 8 | 0.613 | 986.8 | 1551.8 | 255.4 | Pasa |
| 8192 | 8 | 1.800 | 1355.0 | 3486.8 | 256.2 | Pasa |
| 32768 | 8 | 4.157 | 1833.4 | 5840.9 | 257.7 | Pasa |

Pasar significa que se comprobó la forma solicitada, coordenadas finitas y
valores dentro de `[0, 1]`. No mide la calidad del resultado. El pipeline
original usa por defecto 8192 anclas y documenta un rango superior equivalente
a 32768; estas mediciones cubren solo la etapa octree, no el decoder gaussiano
ni la generación de imagen a 3D.

El tiempo comienza después de cargar y evaluar el modelo y la condición e incluye
el muestreo, la sincronización eager y la primera compilación del proceso; la
validación de salida queda fuera del tiempo. El pico activo MLX se mide desde
un reinicio justo antes del muestreo e incluye los aproximadamente 194.3 MiB
del modelo y la entrada activos, pero **excluye buffers en caché**. La caché
final es una instantánea separada, no un pico; no debe sumarse al pico activo
como si se hubieran medido simultáneamente. El pico RSS es el máximo del proceso
en macOS, incluye la preparación y no representa toda la memoria Metal/unificada.
Estos contadores no son una medición completa de la memoria del sistema y pueden
cambiar según entradas, asignador, otras cargas o hardware.

Para precisión numérica, una comparación independiente con aleatoriedad
controlada usó el mismo checkpoint, semilla 101, 32 anclas, 4 niveles y 32 tokens
de condición. Las 96 coordenadas FP32 coincidieron exactamente con PyTorch en
MLX CPU y Metal cuando todos los desplazamientos se fijaron a `0.375`. Es un
caso reducido, no una garantía de precisión a escala completa o con aleatoriedad.

## Reproducir

Obtén los pesos bajo sus condiciones aplicables y mantenlos locales. Cada
comando de perfil inicia un proceso nuevo y solo imprime JSON agregado:

```sh
.venv-reference/bin/python -m scripts.profile_octree \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors \
  --num-points 8192 --level 8 --context-tokens 8192 --device gpu
```

Para el caso de precisión, usa el SDK original fijado en `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_octree_traversal \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors \
  --num-points 32 --level 4 --context-tokens 32 --device gpu
```

Ningún comando publica pesos ni entradas. Los resultados corresponden a este
hardware y configuración; los límites de memoria en otros equipos y la precisión
mixta siguen sin validar.
