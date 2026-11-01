# Carga preliminar de checkpoints FP32 del denoiser

[English](../docs/dit-checkpoints.md)

`idol_mlx.dit_weights.load_dit_checkpoint(model, path)` carga un archivo
`safetensors` **local y confiable** con nombres del denoiser PyTorch original
fijado. Mapea los nombres conocidos de contenedores secuenciales a MLX, valida
los esquemas completos de origen y destino contra el `LatentSeqMMFlowModel`
suministrado y carga estrictamente parámetros y el buffer `pos_pe`. No cambia
tipos ni disposiciones, omite tensores ni acepta formas o tipos incorrectos.
Todos los tensores del modelo y checkpoint deben ser FP32. Comprueba que la
configuración del modelo corresponda al checkpoint antes de cargar.

```python
from idol_mlx.dit_weights import load_dit_checkpoint

# model es un LatentSeqMMFlowModel configurado previamente.
# La ruta es un checkpoint safetensors local, confiable y compatible.
load_dit_checkpoint(model, "/trusted/path/denoiser.safetensors")
```

El cargador se probó con un checkpoint FP32 sintético pequeño de 123 tensores
y cargó estrictamente el checkpoint entrenado de 560 tensores en un
[diagnóstico reducido](dit-trained-validation.md). Ese diagnóstico no superó
las tolerancias numéricas provisionales; cargar no garantiza paridad. Las pruebas comparan la salida del modelo
cargado con PyTorch y comprueban el rechazo de claves ausentes o extra, formas
o tipos incorrectos y modelos de destino no FP32. También verifican que un
checkpoint rechazado no modifique los parámetros. Los checkpoints originales
de precisión mixta no están soportados sin una conversión validada por separado.
Faltan pruebas de memoria a tamaño completo, paridad con pesos entrenados e
inferencia integral. No subas pesos, credenciales ni entradas privadas.

Consulta el [denoiser FP32 preliminar](dit-fp32.md) para pruebas y limitaciones.
