# Captura de tensores sintéticos de referencia

Esta es una referencia parcial para el issue #5. Captura entradas **sintéticas**
y pesos inicializados aleatoriamente de algunos componentes originales; no
incluye checkpoints entrenados, imágenes de entrada, credenciales ni datos
personales. No demuestra equivalencia con MLX ni completa la referencia del
pipeline.

El generador usa el SDK original en el
[commit 4702a7f00850e34b6e83a2170cc123c3e9570744](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744)
y el entorno de referencia en CPU descrito en
[environments.md](../docs/environments.md). Los resultados pueden variar entre
versiones o plataformas: compara ambos backends con el **mismo archivo
capturado**, sin asumir que una semilla igual produce pesos o entradas idénticos.

## Tensores incluidos

| Prefijo | Datos capturados | Alcance |
| --- | --- | --- |
| `repo.*` | Entrada FP32 explícita, salida rotatoria de valores reales y todos los tensores del state dict de un RePo pequeño inicializado aleatoriamente. | Ancho 24, 2 cabezas y 3 tokens. |
| `attention.*` | Entrada, posiciones rotatorias, salida y todos los tensores del state dict de una atención RoPE pequeña inicializada aleatoriamente. | Ancho 24, 2 cabezas y 3 tokens. |
| `sampler.*` | Ruido inicial, condiciones positiva y negativa y salida final explícitos. | Dos pasos Euler, desplazamiento 3 y guidance 3; modelo de prueba de velocidad constante. |
| `camera.*` | Cinco valores de entrada y salida ordenada como yaw, pitch, radio y campo de visión. | Un ejemplo conocido de cámara en perspectiva; salida FP64. |

Cada clave de peso usa `<módulo>.weights.<clave_state_dict>`; las claves de
entrada y salida son explícitas. Solo se cubren estos componentes. La
aleatoriedad del VAE, los pesos entrenados, el DiT, los decodificadores
completos, los encoders y el pipeline completo quedan para trabajo posterior.

## Generación y pruebas

Prepara la copia original y el entorno fijado siguiendo
[environments.md](../docs/environments.md). Confirma que la revisión coincide
con el commit anterior. Desde la raíz del repositorio:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python scripts/capture_reference.py --output /tmp/idol-synthetic-reference.safetensors
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -p 'test_capture_reference.py' -v
```

Usa un destino nuevo: el script no sobrescribe archivos. Genera un archivo
safetensors que ambos backends pueden leer; todos sus pesos son sintéticos. Las
pruebas verifican formas, resultados conocidos del sampler y cámara,
serialización, protección de archivos y reconstrucción de las salidas a partir
de los pesos capturados. Sin `IDOL_REFERENCE_SOURCE` se omiten las pruebas;
eso no constituye validación. Las pruebas no verifican el commit automáticamente.

Validado en CPU con CPython 3.13.14, PyTorch 2.15.0 y safetensors 0.8.0:
las dos pruebas pasaron. Los resultados se generan localmente y no se incluyen
en el repositorio. Revisa cualquier artefacto antes de compartirlo; nunca subas
credenciales, imágenes privadas, rutas personales ni pesos restringidos.

Versión principal: [docs/reference-capture.md](../docs/reference-capture.md).
