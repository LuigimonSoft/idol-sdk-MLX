# Bloques transformer preliminares del decodificador

[English](../docs/decoder-transformers.md)

Los siguientes bloques de `idol_mlx.layers` adaptan el
[SDK original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/layers.py):

- `TransformerCrossBlock` combina atención propia, atención cruzada y una red
  feed-forward con normalización y conexiones residuales.
- `ModulatedTransformerCrossOnlyBlock` usa atención cruzada y una red
  feed-forward, moduladas por seis componentes de desplazamiento, escala y
  compuerta para la decodificación del octree.

Las pruebas sintéticas FP32 comparan cada bloque con PyTorch usando las mismas
entradas, contexto y parámetros. La prueba mapea los nombres secuenciales y
comprueba las formas antes de cargar; no es un convertidor automático de pesos
entrenados. Se prueban longitudes de consulta y contexto distintas, un contexto
modificado y que compuertas en cero preservan la entrada del bloque modulado.
Las tolerancias provisionales son `atol=1e-5, rtol=1e-4`.

No se han validado FP16/BF16, pesos entrenados, todas las formas ni la
integración del modelo completo. Estos bloques no reciben máscaras de atención,
como en la interfaz original. Consulta también la
[atención del decodificador](decoder-attention.md) y el
[transformer unificado](unified-transformer.md).

Con MLX, PyTorch y el SDK original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_layers_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta las pruebas de capas con MLX CPU.
