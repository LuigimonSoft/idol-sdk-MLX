# Validación de configuración y checkpoints

Este cargador inicial valida la estructura común de una configuración local de
Idol y lee checkpoints MLX `.safetensors` según un esquema explícito. No
descarga pesos, no convierte disposiciones de checkpoints PyTorch ni incluye los
esquemas específicos de cada modelo.

## Configuración

`idol_mlx.config.load_config(path)` devuelve un diccionario después de
comprobar:

- `denoiser`, `decoder` y `decoder_gs` son objetos con `name` no vacío y
  `args` como objeto.
- `image_size` y `max_voxel_level` son enteros positivos, no booleanos.
- `image_encoders` contiene cadenas no vacías `dinov3` y `flux2_vae`.

Se conservan campos desconocidos. Estas comprobaciones no validan cada
argumento, la arquitectura admitida, el acceso al repositorio ni la
compatibilidad semántica; cada modelo debe validar sus propios requisitos cuando
se implemente. Una configuración ausente, ilegible, incorrecta o con estructura
inválida produce `ConfigError`.

## Pesos

`idol_mlx.weights.load_weights(path, expected)` lee un archivo local
`.safetensors` con MLX. `expected` es un diccionario no vacío de nombres
completos a `TensorSpec(shape, dtype)`. El cargador rechaza claves ausentes o
inesperadas y cualquier discrepancia de forma o tipo antes de devolver los
arrays. Los tipos usan nombres MLX como `float32` y `bfloat16`. No se
convierten, redimensionan ni transponen tensores silenciosamente.

```python
from idol_mlx.weights import TensorSpec, load_weights

# Ejemplo solamente; no es el esquema de un checkpoint Idol.
expected = {"weight": TensorSpec((2, 3), "float16")}
weights = load_weights("converted-example.safetensors", expected)
```

Los errores de lectura o esquema producen `CheckpointError`; los esquemas
inválidos suministrados por el usuario producen `ValueError`. El llamador debe
proporcionar el esquema completo derivado del modelo correspondiente. Se
requiere la extensión `.safetensors` y las rutas son entradas locales
confiables; esto no es un sandbox para rutas no confiables. La carga no demuestra
procedencia, licencia, integridad de descarga ni equivalencia numérica.

## Validación y trabajo pendiente

Las pruebas unitarias cubren la estructura y errores de claves, formas, tipos y
lectura con entradas sintéticas. También se verificaron en el Mac de referencia
una escritura y lectura reales con MLX de tensores FP16/FP32 y la configuración
pública de Idol. Los nombres reales de checkpoints, esquemas del modelo
entrenado y reglas de conversión quedan pendientes del issue #14. No se
almacenan pesos restringidos ni credenciales en este repositorio.

Versión principal: [docs/config-and-weights.md](../docs/config-and-weights.md).
