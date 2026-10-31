# Entornos de desarrollo reproducibles

El proyecto está en preparación y el pipeline MLX aún no está implementado. Estos
entornos independientes establecen una referencia inicial para macOS ARM64. No
demuestran equivalencia de resultados ni inferencia completa.

## Configuración probada

- Mac mini con Apple M4 y 32 GB de memoria unificada; macOS 27.2 (26B123), arm64.
- CPython 3.13.14 y uv 0.12.1.
- Dependencias de referencia resueltas desde el commit
  [4702a7f00850e34b6e83a2170cc123c3e9570744](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744)
  del SDK original.
- MLX 0.32.3 y mlx-metal 0.32.3; PyTorch 2.15.0, torchvision 0.29.2,
  Transformers 5.22.1 y Diffusers 0.42.1. Los archivos de requisitos incluyen
  todas las dependencias transitivas y hashes de sus artefactos.

Los archivos están destinados a la versión de Python y plataforma probadas.
Otras configuraciones necesitan una nueva resolución y validación; no se afirma
reproducibilidad para otras versiones de parche de Python.

## Instalación

Instala [uv](https://docs.astral.sh/uv/getting-started/installation/) y CPython
3.13.14, por ejemplo con `uv python install 3.13.14`. Desde la raíz:

```sh
uv venv --python 3.13.14 .venv-reference
uv pip sync --python .venv-reference/bin/python --require-hashes environments/reference.txt

uv venv --python 3.13.14 .venv-mlx
uv pip sync --python .venv-mlx/bin/python --require-hashes environments/mlx.txt
```

El archivo de referencia instala las dependencias del commit indicado, pero no
su código fuente. Para comparar con el SDK original, obtén ese commit en un
directorio separado:

```sh
git clone https://github.com/speridlabs/idol-sdk.git ../idol-sdk-reference
git -C ../idol-sdk-reference checkout 4702a7f00850e34b6e83a2170cc123c3e9570744
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -c "import idol; from diffusers.models import AutoencoderKLFlux2; from transformers import AutoModel; from torchvision.ops import deform_conv2d; print('Reference imports OK')"
```

Para comprobar MLX y Metal:

```sh
.venv-mlx/bin/python -c "import mlx.core as mx; assert mx.metal.is_available(); assert mx.default_device() == mx.gpu; x = mx.array([1., 2., 3.]) * 2; mx.eval(x); assert x.tolist() == [2., 4., 6.]; print('Metal OK')"
```

En un entorno aislado o sin acceso gráfico, Metal puede no estar disponible
aunque el equipo lo soporte; ejecuta la prueba en una terminal local normal.
La disponibilidad de MPS en PyTorch se comprueba con
`.venv-reference/bin/python -c "import torch; print(torch.backends.mps.is_available())"`.
El pipeline original selecciona automáticamente CUDA o CPU, pero no MPS; esta
configuración no cambia ese comportamiento.

## Validación y límites

Ambos entornos se instalaron verificando hashes. Las importaciones de referencia
funcionaron; MLX ejecutó el cálculo mostrado en Metal y PyTorch detectó MPS fuera
del aislamiento. No se descargaron pesos de acceso restringido ni se ejecutaron
pruebas de inferencia completa o de equivalencia de resultados. El acceso a los
repositorios de DINOv3 y FLUX.2 y sus condiciones se verifica en otra tarea.

Versión principal en inglés: [docs/environments.md](../docs/environments.md).
