# Instalación del paquete MLX

El paquete `idol_mlx` es una base inicial de empaquetado. Todavía no implementa
el pipeline de inferencia, carga de modelos, CLI ni generación de 3D.

## Configuración de referencia

La configuración inicial probada es Apple Silicon, CPython 3.13.14 y MLX
0.32.3. El proyecto limita Python a la serie 3.13 y fija MLX 0.32.3; otras
combinaciones necesitan validación. La detección y el diagnóstico de Metal se
abordan en el issue #12. Instalar el paquete no demuestra por sí solo que la
máquina pueda acceder a la GPU.

## Instalación desde una copia del repositorio

Desde la raíz, con [uv](https://docs.astral.sh/uv/) y CPython 3.13.14
disponibles:

```sh
uv venv --python 3.13.14 .venv-mlx
uv pip install --python .venv-mlx/bin/python -e .
.venv-mlx/bin/python -c "import idol_mlx; print(idol_mlx.__version__)"
```

La instalación editable resuelve las dependencias desde el índice configurado.
La referencia MLX con hashes se documenta por separado en
[environments.md](../docs/environments.md); ese archivo no instala el proyecto.

La dependencia del runtime MLX no instala PyTorch; el entorno PyTorch de
referencia es independiente. No se incluyen pesos ni credenciales. No agregues
tokens, imágenes privadas ni pesos generados al repositorio.

## Verificación

Se construyó un wheel y se instaló en un entorno virtual limpio del Mac Apple
Silicon de referencia con CPython 3.13.14. Desde fuera de la copia del código,
la importación y un cálculo MLX simple en Metal funcionaron sin importar
PyTorch. Es una prueba de empaquetado, no de inferencia ni equivalencia.

Versión principal: [docs/mlx-package.md](../docs/mlx-package.md).
