# Idol SDK para MLX

Adaptación en desarrollo para Apple Silicon de [Idol SDK](https://github.com/speridlabs/idol-sdk),
que convierte una imagen en un objeto 3D de Gaussian splatting.

**Estado: desarrollo inicial.** El pipeline MLX de imagen a 3D, la conversión
de modelos y la generación PLY todavía no están implementados. La instalación
indicada a continuación no realiza inferencia.

[English](../README.md)

## Disponible actualmente

- Una base instalable del paquete `idol_mlx` para la configuración probada de
  Python 3.13 y MLX 0.32.3.
- Un diagnóstico de Metal que prueba una operación pequeña en GPU sin cambiar
  el dispositivo predeterminado de MLX.
- Validación de la configuración común y un cargador local de safetensors con
  MLX que comprueba un esquema de tensores proporcionado explícitamente.
- Mapeo explícito de nombres en memoria y conversión de disposición para Conv2d
  estándar sin grupos, con validación estricta del esquema de salida.
- Una política preliminar opcional de precisión y límites de evaluación explícitos.
- Capas MLX de normalización, feed-forward y embeddings escalares, espaciales y rotatorios,
  con pruebas sintéticas frente a PyTorch.
- Atención propia y cruzada preliminar del decodificador y atención rotatoria
  del denoiser, con pruebas sintéticas.
- Bloques transformer preliminares unificado y del decodificador con pruebas sintéticas FP32.
- Un sampler Euler preliminar con guía classifier-free opcional y pruebas sintéticas.
- Ruido MLX con semilla y comparaciones sintéticas de trayectoria por paso.
- Decodificación MLX del latente de cámara a los campos compatibles con JSON del SDK original.
- Exportación binaria PLY 3DGS con las convenciones originales de coordenadas y atributos.
- Conversión FP32 de atributos gaussianos según las convenciones originales de posición, escala, rotación, opacidad y SH.
- Decoder FP32 preliminar de características gaussianas con carga estricta;
  algunos casos reducidos con pesos entrenados aún fallan las tolerancias en Metal;
  hay diagnósticos opcionales de bloques y atención.
- Recorrido preliminar octree y muestreo de anclas con aleatoriedad MLX controlada.
- Una primitiva de muestreo sistemático octree con desplazamientos aleatorios explícitos.
- Red preliminar FP32 de probabilidades octree y mapeo estricto de checkpoints,
  probados con pesos sintéticos y escenarios reducidos con checkpoint entrenado.
- Un denoiser FP32 preliminar y carga estricta de checkpoints. Un diagnóstico
  reducido con pesos entrenados supera las tolerancias provisionales para varias entradas FP32.
- Una herramienta preliminar de comparación de archivos con tolerancias explícitas.
- Capturas sintéticas y pruebas de algunos componentes originales, además de
  métricas preliminares de equivalencia numérica.

Las rutas FP32 preliminares del denoiser y los decoders tienen validación
limitada con datos sintéticos y checkpoints entrenados. Quedan pendientes más
esquemas y conversiones de checkpoints, encoders, decodificadores completos e
inferencia integral.

## Instalación

En Apple Silicon, con CPython 3.13.14 y
[uv](https://docs.astral.sh/uv/) disponibles:

```sh
uv venv --python 3.13.14 .venv-mlx
uv pip install --python .venv-mlx/bin/python -e .
.venv-mlx/bin/python -c "import idol_mlx; print(idol_mlx.__version__)"
```

Comprueba Metal por separado:

```sh
.venv-mlx/bin/python -c "from idol_mlx.device import require_metal; require_metal(); print('Metal check passed')"
```

Consulta la [instalación del paquete MLX](mlx-package.md) y el
[diagnóstico de Metal](metal-diagnostics.md). Metal puede no estar disponible
en sesiones restringidas o sin interfaz gráfica aunque el equipo sea compatible.

## Pruebas y acceso a modelos

La suite completa usa un entorno de referencia independiente y una copia del
SDK original fijada a un commit. Sigue
[entornos de desarrollo](environments.md) y
[pruebas de componentes de referencia](reference-components.md) antes de
ejecutar:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -v
```

Si `IDOL_REFERENCE_SOURCE` no está definido, se omiten las pruebas que
dependen del SDK original; una ejecución con pruebas omitidas no constituye una
validación completa. Las pruebas actuales no demuestran inferencia completa ni
equivalencia PyTorch/MLX. Un [diagnóstico reducido con checkpoint entrenado](dit-trained-validation.md)
compara las salidas del denoiser con frecuencias nativas y derivadas de la
referencia; ambas rutas pasan en los escenarios FP32 reducidos registrados.

La futura inferencia completa requiere acceso aprobado a DINOv3 y FLUX.2 dev y
el cumplimiento de sus licencias respectivas. Consulta
[acceso y licencias de modelos](model-access.md). No subas credenciales,
imágenes privadas ni pesos al repositorio.

## Documentación

- [Exportación PLY de gaussianas](gaussian-ply.md)

- [Conversión de atributos gaussianos](gaussian-attributes.md)

- [Decoder de características gaussianas](gaussian-decoder.md)

- [Observaciones de límites y memoria del octree](octree-profiling.md)

- [Muestreo de anclas octree](octree-traversal.md)

- [Muestreo sistemático octree](octree-resampling.md)

- [Red de probabilidades octree](octree-network.md)

- [Decodificación de cámara](camera-decoding.md)

- [Arquitectura y tensores del original](architecture.md)
- [Arquitectura MLX propuesta](mlx-architecture.md)
- [Validación de configuración y checkpoints](config-and-weights.md)
- [Mapeo preliminar de pesos](weight-mapping.md)
- [Política de precisión y evaluación diferida](precision-evaluation.md)
- [Capas de normalización MLX](normalization-layers.md)
- [Capas feed-forward MLX](feedforward-layers.md)
- [Embedding sinusoidal escalar](sinusoidal-embedding.md)
- [Embeddings espaciales](spatial-embeddings.md)
- [Rotación y RePo 3D](rotary-repo.md)
- [Atención del decodificador](decoder-attention.md)
- [Atención rotatoria del denoiser](rotary-attention.md)
- [Bloque transformer unificado](unified-transformer.md)
- [Bloques transformer del decodificador](decoder-transformers.md)
- [Sampler Euler](euler-sampler.md)
- [Ruido con semilla y reproducibilidad](reproducibility.md)
- [Trayectorias sintéticas del sampler](sampler-trajectories.md)
- [Denoiser FP32 preliminar](dit-fp32.md)
- [Carga de checkpoints FP32 del denoiser](dit-checkpoints.md)
- [Diagnóstico con checkpoint entrenado](dit-trained-validation.md)
- [Capturas sintéticas de referencia](reference-capture.md)
- [Métricas y tolerancias provisionales](parity-metrics.md)
- [Comparación de salidas entre backends](parity-harness.md)
- [Imágenes de prueba reproducibles](test-images.md)
- [CI y SonarCloud](ci-sonarcloud.md)
- [Límite de confianza de rutas de salida](output-path-security.md)
