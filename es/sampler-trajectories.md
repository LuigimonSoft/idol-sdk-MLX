# Trayectorias sintéticas del sampler

[English](../docs/sampler-trajectories.md)

`scripts.sampler_trajectory` compara cada estado previo a la actualización y el
estado final de un sampler sintético determinista frente a PyTorch original
fijado. Ambos backends reciben el mismo ruido y condiciones FP32 explícitos. La
velocidad sintética depende de la muestra, el tiempo y la condición; la fuerza
de guía y el desplazamiento temporal son 3. Se admiten guía separada y por
lotes. Estas ejecuciones no validan el denoiser entrenado, el resultado 3D,
operaciones aleatorias del modelo ni rendimiento de producción.

Instala el entorno de referencia fijado y MLX 0.32.3 en el mismo entorno;
define `PYTHONPATH` con la copia del SDK original fijado. Desde la raíz:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.sampler_trajectory --steps 50 --batch-cfg
```

Usa `--steps 2` para una secuencia corta u omite `--batch-cfg` para llamadas
separadas. `--atol` y `--rtol` configuran tolerancias (por defecto `1e-5` y
`1e-4`). El informe JSON incluye una entrada por estado, incluidos el inicial y
el final, con errores absolutos máximo y medio, número de fallos y resultado,
además de versiones MLX y PyTorch, dispositivo MLX, tipo y tolerancias. Si
necesitas conservarlo, redirige la salida a una ruta local confiable; no subas
entradas privadas ni pesos. El código de salida es 0 si todos los estados pasan
y 1 si una comparación numérica falla.

La prueba unitaria cubre trayectorias de 2 y 50 pasos en ambos modos y comprueba
que un estado modificado se detecta como fallo. CI la ejecuta en MLX CPU. Una
prueba local en Metal de 50 pasos con guía por lotes obtuvo 51 estados correctos
y diferencia absoluta máxima `0.0` para este modelo sintético concreto; no es
una garantía general de precisión.
