# Embedding sinusoidal escalar preliminar

[English](../docs/sinusoidal-embedding.md)

`idol_mlx.layers.SinusoidalEmbedder` adapta el embedding de tiempo o nivel del
[SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
Construye frecuencias FP32 espaciadas exponencialmente, concatena características
coseno y seno, opcionalmente multiplica los ángulos por `2*pi` y aplica dos
capas lineales con sesgo y SiLU entre ellas. La entrada es un lote
unidimensional de escalares, que se convierten a FP32 antes del cálculo. La
adaptación conserva la disposición original y no carga checkpoints automáticamente.

Las pruebas sintéticas copian pesos y sesgos FP32 del módulo PyTorch y comparan
salidas con entradas compartidas, incluidas entradas negativas y cero, tipos
FP32/FP16, dimensiones de frecuencia 8 y 10, dos periodos y ambas opciones de
`two_pi`. La tolerancia es `atol=1e-5, rtol=1e-4`. No validan checkpoints
entrenados, pesos de precisión reducida, BF16, todos los rangos numéricos ni la
inferencia completa. Otros embeddings temporales y espaciales siguen pendientes.

Con MLX, PyTorch y el SDK fijado instalados, ejecuta:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX Metal cuando esté disponible; PyTorch
corre en CPU. El paso existente de CI para capas compartidas ejecuta las
mismas pruebas con MLX CPU.
