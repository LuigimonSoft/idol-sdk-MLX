# Capas preliminares feed-forward en MLX

[English](../docs/feedforward-layers.md)

`idol_mlx.layers.FeedForwardNet` y `MLP` reproducen la estructura de capas
lineales y GELU del [SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
Utilizan capas lineales MLX con sesgo y `mlx.nn.gelu_approx`, correspondiente a
`GELU(approximate="tanh")` de PyTorch. Los pesos lineales tienen forma
`(salidas, entradas)`. No se mapean checkpoints automáticamente: los nombres
secuenciales originales como `mlp.0.weight` difieren de nombres MLX como
`mlp.layers.0.weight`; usa reglas explícitas y valida el esquema de destino.

Cuando el `MLP` original tiene una sola capa, usa `inner_channels` como tamaño
de entrada de la capa final aunque difiera de `channels`. Esta adaptación
conserva ese comportamiento; la última dimensión de entrada debe coincidir.

Las pruebas sintéticas copian pesos y sesgos inicializados de PyTorch y usan las
mismas entradas en FP32 y FP16. Cubren dos proporciones de expansión y MLP de
una, dos y tres capas, con varios lotes y tokens. Las tolerancias exploratorias
son `atol=1e-5, rtol=1e-4` para FP32 y `atol=2e-3, rtol=2e-3` para FP16.
No validan checkpoints entrenados, BF16, todos los rangos de entrada ni la
calidad completa. Los bloques originales con SiLU no están cubiertos aquí.

Con el entorno combinado de referencia y MLX y la ruta al SDK fijado:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para ejecutar MLX en Metal cuando esté
disponible; PyTorch permanece en CPU. CI ejecuta estas pruebas con MLX CPU tras
instalar la versión fijada. Consulta [capas de normalización](normalization-layers.md)
para los otros bloques compartidos implementados.
