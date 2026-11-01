# Atención preliminar del decodificador

[English](../docs/decoder-attention.md)

`idol_mlx.layers.MultiHeadAttention` adapta la atención propia y cruzada del
[SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
Las proyecciones de consultas y claves usan la normalización original por
cabeza, y la atención escalada usa `1/sqrt(head_dim)` y una proyección de
salida. La atención cruzada recibe un contexto separado, con longitud y número
de características propios. Esta capa, al igual que la original, no recibe
máscaras; la atención con máscaras no se valida aquí. La carga de checkpoints
no es automática.

Las pruebas sintéticas copian pesos, sesgos y escalas no triviales de consultas
y claves desde PyTorch tras comprobar las formas. Usan lote 2, longitud de
consulta 3, longitud de contexto 5, dos cabezas y FP32/FP16; también comparan
un contexto modificado. Las tolerancias provisionales son `atol=1e-5,
rtol=1e-4` en FP32 y `atol=3e-3, rtol=3e-3` en FP16. Estos casos no validan
todas las formas, máscaras, BF16, pesos entrenados, atención rotatoria del
denoiser ni inferencia completa.

Con MLX, PyTorch y el SDK original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta estas pruebas de paridad con MLX CPU.
