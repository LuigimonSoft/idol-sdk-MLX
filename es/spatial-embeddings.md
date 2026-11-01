# Embeddings espaciales preliminares

[English](../docs/spatial-embeddings.md)

Los dos embeddings de coordenadas de `idol_mlx.layers` reproducen el
[SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py):

- `PcdAbsolutePositionEmbedder` codifica coordenadas por token usando las
  frecuencias de potencias de dos del denoiser, incluida la regla de frecuencias
  extendidas; las entradas FP32/FP16 probadas producen salida FP32.
- `PcdAbsolutePositionEmbedderv2` codifica coordenadas por muestra usando las
  frecuencias logarítmicas del decodificador y el tipo de entrada.

Ambos concatenan seno y coseno en el orden original y rellenan con ceros hasta
el número de canales solicitado cuando es necesario. Para `freq_dim=170,
max_res=16`, se usa una constante FP32 canónica generada con el embedding del
SDK original fijado en PyTorch 2.15.0 CPU; no contiene datos del checkpoint.
Evita diferencias de redondeo entre backends amplificadas por la trigonometría
de alta frecuencia. Otras configuraciones de V1 usan el cálculo general de MLX. No
se agregan parámetros entrenables, dependencia de PyTorch para inferencia ni
carga de checkpoints. Los valores se reproducen con
`PcdAbsolutePositionEmbedder(1024, in_channels=3, max_res=16).freqs` en el
entorno de referencia fijado; la prueba compara cada valor de forma exacta.
Para V2 con `freq_dim=170,
max_res=10`, las frecuencias canónicas del decoder se generan igualmente con
`PcdAbsolutePositionEmbedderv2(1024, in_channels=3, max_res=10).freqs` y se
convierten al tipo de entrada como en la referencia; otras configuraciones V2
usan el cálculo general de MLX. El decoder gaussiano también
usa V2; este cambio no valida la paridad de esa red completa.

Las pruebas sintéticas de paridad usan las mismas coordenadas en PyTorch y MLX.
V1 cubre lotes con varios tokens, relleno, frecuencias extendidas y entradas
FP32/FP16, con `atol=1e-5, rtol=1e-4`. V2 cubre varias muestras, relleno, exponentes fraccionarios de frecuencias
intermedias y entradas FP32/FP16; FP32 usa las mismas tolerancias y FP16 usa `atol=3e-3,
rtol=3e-3`. Estos casos y tolerancias provisionales no validan BF16, todas las
formas, todos los modelos entrenados ni el pipeline completo.

Con MLX, PyTorch y el código original fijado disponibles:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch corre en CPU. CI ejecuta este archivo de pruebas con MLX CPU. Consulta
también el [embedding sinusoidal escalar](sinusoidal-embedding.md).
