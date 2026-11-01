# Bloque transformer unificado preliminar

[English](../docs/unified-transformer.md)

`idol_mlx.layers.UnifiedTransformerBlock` adapta el bloque del denoiser y
refinador del [SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py).
Combina LayerNorm32, atención propia rotatoria real, un bloque feed-forward y
conexiones residuales. Con modulación activada suma una tabla entrenable y
divide el resultado en seis componentes de desplazamiento, escala y compuerta.
Sin modulación utiliza normalización afín y actualizaciones residuales directas.

Las pruebas sintéticas FP32 copian cada parámetro desde PyTorch tras comprobar
nombres y formas, mapeando explícitamente los nombres secuenciales a MLX. Se
comparan ambas modalidades con entradas, pares rotatorios y pesos idénticos.
La prueba modulada comprueba que una modulación no nula cambie el resultado y
que compuertas efectivas en cero conserven la entrada. La tolerancia es
`atol=1e-5, rtol=1e-4`. Este mapeo de prueba no convierte checkpoints entrenados.

El bloque no se ha validado con FP16/BF16, pesos entrenados, salida generada
por RePo, todas las formas ni el modelo completo. Los demás tipos de bloques
transformer siguen pendientes.

Con MLX, PyTorch y el código original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta las pruebas de capas compartidas con MLX CPU.
