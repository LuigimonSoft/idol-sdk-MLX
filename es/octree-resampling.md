# Primitiva de muestreo sistemático octree

[English](../docs/octree-resampling.md)

`idol_mlx.octree_sampling.systematic_resample` distribuye el conteo de cada
padre entre sus hijos mediante muestreo sistemático. Recibe arrays MLX de
probabilidades `(..., bins)`, conteos enteros `(...)` y desplazamientos FP32
explícitos `(...)` en `[0, 1)`. Devuelve un array int64 `(..., bins)` cuya suma
en el último eje es el conteo de entrada de cada fila. Las celdas octree usan
8 grupos, pero la primitiva admite cualquier número positivo de grupos.

```python
import mlx.core as mx
from idol_mlx.octree_sampling import systematic_resample

probabilities = mx.array([[0.1, 0.2, 0.3, 0.4]])
counts = mx.array([10])
offsets = mx.random.uniform(shape=(1,), key=mx.random.key(42))
allocation = systematic_resample(probabilities, counts, offsets=offsets)
```

El llamador proporciona desplazamientos independientes y controla la clave
aleatoria. Una semilla MLX no reproduce la secuencia de PyTorch; para comparar
las implementaciones usa los mismos desplazamientos. La función limita a cero
las probabilidades negativas, divide por `max(suma_fila, 1)` y usa una fila
uniforme cuando la suma es cero, igual que la [referencia fijada](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
Para sumas menores que uno, la masa restante se asigna al último grupo, como en
la referencia; quien necesite una distribución normalizada debe proporcionarla.

La validación sincroniza la evaluación MLX y rechaza valores no finitos,
conteos no enteros, formas inválidas, desplazamientos fuera de `[0, 1)` y
conteos superiores a `2**24 - 1`; el límite acota la malla FP32 y las iteraciones
de búsqueda. La implementación evita asignar una malla por punto y usa una
búsqueda binaria fija de 24 pasos por grupo. Aún no se ha medido su rendimiento
con octrees de tamaño completo.

Las pruebas comparan desplazamientos controlados con PyTorch original, incluidos
límites, filas cero/negativas/no normalizadas, conteos variados y lotes. Una
prueba de 4096 filas comprueba desplazamientos independientes, conteos exactos
frente a la referencia y frecuencias empíricas. CI ejecuta las pruebas con MLX
CPU; también se ejecutaron localmente con Metal. Esta es solo la primitiva de
distribución: faltan el recorrido del árbol de tamaño variable, posiciones
aleatorias dentro de hojas, conteo completo de anclas e inferencia integral
para el issue #32.
