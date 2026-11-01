# Ruido con semilla y reproducibilidad

[English](../docs/reproducibility.md)

`idol_mlx.sampler.seeded_noise` genera ruido normal estándar usando una clave
local explícita de MLX. Acepta una tupla no vacía de dimensiones positivas, una
semilla entera sin signo de 64 bits y el tipo `float32`, `float16` o `bfloat16`.
No cambia la semilla ni avanza la secuencia aleatoria global de MLX. Pasa su
salida directamente a `sample_euler` o proporciona un tensor MLX explícito ya
existente como ruido inicial.

```python
from idol_mlx.sampler import sample_euler, seeded_noise

noise = seeded_noise((2, 16), 42)
result = sample_euler(model, noise, condition, steps=4)
```

Con la misma versión de MLX, dispositivo, forma, tipo, modelo, parámetros,
entradas y opciones, las pruebas sintéticas FP32 repetidas producen ruido y
salidas idénticos. Esto no promete igualdad bit a bit entre versiones de MLX,
equipos, dispositivos, precisiones o modelos. MLX y PyTorch no necesariamente
generan los mismos valores con la misma semilla; para comparar frameworks usa
el mismo tensor explícito capturado. Otras operaciones aleatorias del modelo
requieren su propio control de aleatoriedad.

Las pruebas cubren semillas iguales y distintas, tipos, entradas inválidas,
aislamiento del generador global y muestreo repetido con un modelo sintético
determinista. No se han probado modelos reales, su aleatoriedad ni trayectorias
completas BF16/FP16.

Con MLX, PyTorch y el SDK original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_sampler_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta las pruebas del sampler con MLX CPU.
