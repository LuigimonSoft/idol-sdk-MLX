# Operaciones rotatorias y RePo 3D preliminares

[English](../docs/rotary-repo.md)

`idol_mlx.layers` adapta la ruta de valores reales de
[`apply_rotary_emb` y `RePo3DRotaryEmbedding`](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py)
del SDK original fijado. RePo predice desplazamientos 3D por cabeza usando
LayerNorm, SiLU y tres capas lineales sin sesgo; vectores de frecuencia
entrenables los convierten en ángulos y pares `(cos, sin)`. La rotación trata
características adyacentes como pares reales e imaginarios y restaura el tipo de
entrada. Las frecuencias pueden compartirse entre cabezas mediante broadcasting.

La adaptación MLX devuelve solo pares reales de coseno y seno; no implementa la
representación compleja del original. Los vectores de frecuencia y pesos
lineales son parámetros y deben cargarse con mapeo y esquema explícitos; no se
incluye la conversión de checkpoints entrenados. La expresión de `clamp_mul`
conserva la estructura original de detención de gradiente, pero este trabajo no
valida gradientes de entrenamiento.

Las pruebas sintéticas comparan con PyTorch usando los mismos pesos y
frecuencias para dos tamaños de cabeza, y prueban rotación con frecuencias
compartidas o por cabeza y entradas FP32/FP16. Las tolerancias provisionales son
`atol=1e-5` y `rtol=1e-4`. Los casos no validan todas las formas, BF16, pesos
entrenados, integración con atención ni inferencia completa.

Con MLX, PyTorch y el SDK original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para probar MLX en Metal cuando esté disponible;
PyTorch corre en CPU. CI ejecuta las pruebas de capas compartidas con MLX CPU.
