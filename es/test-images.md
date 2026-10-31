# Imágenes de prueba reproducibles

Este pequeño conjunto público sirve para el preprocesamiento y futuras
comparaciones funcionales y visuales de extremo a extremo. No contiene imágenes
personales ni pesos de modelos.

| Archivo | Tipo | Propósito | Fuente y derechos |
| --- | --- | --- | --- |
| `coffee.png` | Fotografía RGB, 600 × 400 | Fotografía real con una taza destacada y fondo. | [scikit-image v0.25.2](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py), fotógrafa Rachel Michetti; el origen la identifica como CC0. |
| `clock_motion.png` | Fotografía en escala de grises, 400 × 300 | Objeto real con desenfoque de movimiento; comprueba escala de grises. | [scikit-image v0.25.2](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py), fotógrafo Stefan van der Walt; el origen indica que la cedió al dominio público. |
| `synthetic_opaque.png` | RGB, 256 × 256 | Objeto sintético opaco; necesita eliminación del fondo. | Generado localmente por el script. |
| `synthetic_alpha.png` | RGBA, 256 × 256 | Objeto sintético con píxeles transparentes y opacos; ejercita el uso de alfa existente. | Generado localmente por el script. |

Las fotografías tienen fondos complejos: son entradas útiles, no máscaras de
referencia ni garantía de calidad de reconstrucción. El pipeline original está
pensado para imágenes de un solo objeto; la escena del café puede presentar
ambigüedad. Las salidas visuales deben revisarse por separado.

## Preparación y pruebas

Prepara el entorno de referencia fijado según
[environments.md](../docs/environments.md). Elige un destino nuevo y vacío
**fuera del repositorio** y ejecuta:

```sh
.venv-reference/bin/python scripts/prepare_test_images.py --output /tmp/idol-test-images
.venv-reference/bin/python -m unittest discover -s tests -p 'test_prepare_test_images.py' -v
```

El generador descarga las dos fotos públicas desde la etiqueta indicada y
verifica sus hashes SHA-256 antes de escribir archivos. Dibuja las imágenes
sintéticas con la versión fijada de Pillow. No sobrescribe nombres existentes;
usa un destino nuevo para repetir. La descarga requiere red, pero las pruebas
unitarias simulan las descargas y funcionan sin conexión. Investiga un fallo de
red o de hash en vez de ignorarlo.

Las pruebas comprueban modos y alfa, rechazo de un hash incorrecto, creación de
los cuatro archivos y protección contra sobrescritura. En el entorno de
referencia pasaron las tres pruebas y una descarga real produjo los tamaños y
modos indicados. Una imagen determinista no garantiza inferencia determinista.

Versión principal: [docs/test-images.md](../docs/test-images.md).
