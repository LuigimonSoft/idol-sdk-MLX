# Muestreo de anclas octree

[English](../docs/octree-traversal.md)

`idol_mlx.octree_sampling.sample_octree` recorre el octree usando un modelo que
devuelve ocho logits por celda, distribuye conteos mediante muestreo sistemático
y agrega un desplazamiento uniforme dentro de cada hoja. Devuelve un array MLX
FP32 `[batch, num_points, 3]` con coordenadas normalizadas. El modelo recibe
centros `[B, cells, 3]`, resoluciones enteras `[B]` y condiciones FP32
`[B, tokens, channels]`. Se puede usar la red octree preliminar como modelo.

```python
from idol_mlx.octree_sampling import sample_octree

# model es un OctreeProbabilityFixedlenDecoder cargado; cond es un array MLX FP32.
anchors = sample_octree(model, cond, num_points=32, level=2, seed=101)
```

`num_points` debe estar en `[0, 2**24 - 1]`, `level` en `[0, 23]` y `seed` ser
un entero sin signo de 64 bits. Cero puntos devuelve un array vacío. Una clave
local MLX controla el muestreo y los desplazamientos; con la misma semilla,
entradas, dispositivo y versión MLX el resultado es reproducible, pero MLX y
PyTorch no comparten secuencia aleatoria. Matemáticamente las coordenadas están
en `[0, 1)`; el redondeo FP32 cerca del extremo superior puede producir `1.0`
en niveles profundos.

Los hijos no vacíos se compactan por lote mediante ordenamiento estable. En cada
nivel se lee al host el mayor número de celdas activas; se usan lotes rellenados
y la expansión de hojas recorre los elementos del lote. Es una implementación
preliminar y eager, no una operación completamente residente en GPU. Los límites
de nivel y puntos no garantizan memoria suficiente; consulta los [escenarios
medidos](octree-profiling.md) y sus límites en un equipo.

Las pruebas controladas comparan el muestreador PyTorch fijado con los mismos
desplazamientos para varios lotes y niveles 0–3. También comprueban conteos,
ocupación uniforme de la raíz, reproducibilidad de semilla y validación de
entradas en CPU y Metal. Con el checkpoint octree de `speridlabs/idol` revisión
`2731c2291fc65a797c5cb627a5e5526fc1490902`, referencia FP32, una entrada de
condición sintética, 32 puntos, nivel 2 y desplazamientos fijados a `0.375`, las
96 coordenadas coincidieron exactamente en CPU y Metal. Este caso no valida
entradas entrenadas arbitrarias, secuencias aleatorias, precisión mixta ni
la generación completa de imagen a 3D. No se publicaron pesos ni entradas.

Con el SDK original fijado en `PYTHONPATH` y archivos locales confiables:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_octree_traversal \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder.safetensors --device gpu
```

Usa `--device cpu` para MLX CPU. El diagnóstico fija los desplazamientos solo
dentro del proceso, muestra métricas agregadas de coordenadas y devuelve 0 solo
si todas coinciden exactamente. Carga los pesos completos del decoder entrenado.
