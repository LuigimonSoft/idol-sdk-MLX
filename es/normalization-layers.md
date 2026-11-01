# Capas preliminares de normalización MLX

[English](../docs/normalization-layers.md)

`idol_mlx.layers` adapta dos capas compartidas del SDK original en el
[commit 4702a7f](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py):

- `LayerNorm32`: convierte la entrada a float32, normaliza la última dimensión
  y restaura el tipo original. Admite la dimensión de características y los
  parámetros afines opcionales usados por Idol; MLX requiere epsilon positivo.
- `MultiHeadRMSNorm`: calcula la norma L2 por cabeza en float32 con denominador
  mínimo `1e-12`, multiplica por `gamma` entrenable y `sqrt(dim)` y restaura el
  tipo de entrada. Es la fórmula original, no un RMSNorm genérico.

Las pruebas sintéticas comparan con las clases PyTorch originales usando las
mismas entradas y parámetros, con y sin parámetros afines, gamma no trivial,
ceros, valores cercanos a cero, FP32 y FP16. Las tolerancias exploratorias son
`atol=1e-5, rtol=1e-4` para FP32 y `atol=1e-3, rtol=1e-3` para FP16; no son
criterios de aceptación del modelo completo. Las pruebas se limitan a esos casos
y no validan pesos entrenados ni BF16.

Con MLX y el entorno de referencia fijado instalados, ejecuta desde la raíz:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para ejecutar MLX en Metal cuando esté
disponible; PyTorch permanece en CPU. CI instala la versión fijada de MLX tras
la suite estándar y ejecuta estas pruebas con MLX CPU. Las capas lineales,
activaciones, otras operaciones y la inferencia completa siguen pendientes.
