# Denoiser FP32 preliminar

[English](../docs/dit-fp32.md)

`idol_mlx.dit.LatentSeqMMFlowModel` implementa una pasada preliminar FP32
adaptada del [denoiser original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/dit.py).
Combina embeddings temporales y de posición, refinadores de ruido y contexto,
rotaciones reales RePo entrenables, bloques transformer conjuntos y salidas de
latente y cámara. Las entradas son diccionarios con arrays `latent`, `camera`,
`feature1` y `feature2`; las salidas contienen `latent` y `camera`. El buffer
posicional `pos_pe` forma parte del estado y debe contener posiciones adecuadas.

Esta implementación es **solo FP32 para la configuración probada**. El original
utiliza FP16/FP32 mixto por defecto. La prueba sintética convierte explícitamente
el modelo PyTorch y su tipo interno a FP32; no valida la precisión predeterminada
del original. `model_channels` debe ser igual a `num_heads * num_head_channels`.

Una prueba de modelo pequeño mapea los 123 parámetros y buffers sintéticos de
PyTorch, verifica cada forma y los carga estrictamente en MLX. Compara las
salidas de latente y cámara para dos tamaños de lote y condición en CPU o Metal,
con tolerancia `atol=2e-5, rtol=2e-4`. Este mapeo solo existe en la prueba;
quedan sin validar la conversión automática de checkpoints, los esquemas por
modelo, pesos entrenados, memoria a tamaño completo e inferencia integral. No
subas pesos, entradas privadas ni credenciales.

Con MLX, PyTorch y el código original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_dit_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta esta prueba sintética con MLX CPU. El
código original atribuye la arquitectura a los equipos Wan y Hugging Face bajo
Apache-2.0; el archivo de código conserva ese aviso.
