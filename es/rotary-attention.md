# Atención rotatoria preliminar del denoiser

[English](../docs/rotary-attention.md)

`idol_mlx.layers.RopeMultiHeadAttention` adapta la atención propia del
[SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
Proyecta consultas, claves y valores; rota consultas y claves con pares reales
de coseno y seno; normaliza por cabeza; calcula atención escalada y proyecta
el resultado. La escala es `1/sqrt(head_dim)`. Como el bloque original, esta
interfaz no acepta máscaras de atención.

El argumento rotatorio utiliza la representación real descrita en
[rotación y RePo](rotary-repo.md); la representación compleja original no está
implementada. Los pesos no se cargan ni convierten automáticamente.

Las pruebas sintéticas usan pesos idénticos y escalas no triviales de
normalización de consultas y claves en PyTorch y MLX. Comparan entradas
FP32/FP16, una y dos cabezas, rotaciones distintas por token y rotaciones
identidad. También comprueban que las rotaciones no triviales cambian la salida
PyTorch. Las tolerancias provisionales son `atol=1e-5, rtol=1e-4` para FP32 y
`atol=3e-3, rtol=3e-3` para FP16. No validan BF16, máscaras, checkpoints
entrenados, todas las formas, la integración de RePo con atención ni el modelo
completo.

Con MLX, PyTorch y el SDK original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta estas pruebas de capas con MLX CPU.
