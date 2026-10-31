# Mapeo preliminar de tensores

[English](../docs/weight-mapping.md)

`idol_mlx.conversion` ofrece un mapeo explícito, uno a uno y en memoria de
los tensores cargados previamente. Cada regla indica el nombre de origen, el
de destino y selecciona `identity` o `conv2d_oihw_to_ohwi`. La segunda opción
transpone un kernel Conv2d estándar desde el orden salida-entrada-alto-ancho de
PyTorch al orden salida-alto-ancho-entrada de MLX. Quien llama debe identificar
las capas compatibles; las convoluciones con grupos, transpuestas y otras
variantes no se han validado. Consulta la documentación de
[PyTorch Conv2d](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv2d.html)
y [MLX conv2d](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html).

```python
from idol_mlx.conversion import TensorMapping, convert_tensors
from idol_mlx.weights import TensorSpec

# tensors contiene un array MLX ya cargado bajo "conv.weight".
converted = convert_tensors(
    tensors,
    [TensorMapping("conv.weight", "weight", "conv2d_oihw_to_ohwi")],
    {"weight": TensorSpec((2, 3, 3, 4), "float32")},
)
```

Este ejemplo espera un kernel de origen con forma `(2, 4, 3, 3)`. Cada nombre
de entrada y salida debe aparecer exactamente una vez y cada resultado debe
coincidir con la forma y el tipo indicados. La conversión no cambia tipos ni
deduce esquemas, nombres o semántica de capas. Quien llama debe proporcionar
entradas confiables y un esquema derivado del modelo de destino real.

Se comparó una convolución sintética sin grupos de tipo float32 entre PyTorch
en CPU y MLX en Metal; la diferencia absoluta máxima de salida fue de
aproximadamente `1.91e-6`. Un único ejemplo no garantiza la equivalencia del
modelo completo. Quedan pendientes los mapeos específicos de cada modelo, la
validación con pesos entrenados, la exportación a archivos y el convertidor
completo.
