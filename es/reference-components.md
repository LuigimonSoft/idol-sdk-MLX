# Pruebas de componentes de referencia

El pipeline original completo aún no se ha ejecutado: no había un token de
Hugging Face disponible durante la comprobación de acceso y se denegaron las
solicitudes sin autenticar a DINOv3 y al VAE de FLUX.2 dev. Consulta
[acceso a modelos y licencias](model-access.md). Se cumple el criterio del issue
de documentar el bloqueo, pero siguen pendientes los resultados completos, las
formas con pesos autorizados y la equivalencia numérica.

Como referencia provisional, `tests/test_reference_components.py` ejecuta casos
pequeños y deterministas con la implementación original. Estas pruebas no
cargan pesos, no ejecutan redes neuronales ni demuestran equivalencia con MLX.

## Configuración probada

- Código original: commit
  [4702a7f00850e34b6e83a2170cc123c3e9570744](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744).
- Entorno de referencia: CPython 3.13.14, PyTorch 2.15.0 y dependencias fijadas
  en `environments/reference.txt`; consulta
  [configuración de entornos](environments.md).
- Dispositivo: CPU, tensores de entrada FP32. Las pruebas no necesitan Metal,
  MPS, pesos, credenciales ni red.
- Resultado: 3 pruebas aprobadas.

## Casos y resultados esperados

1. Un modelo de prueba de velocidad constante recibe condición 2, condición
   negativa 1, valor inicial 10, guidance 3 y dos pasos Euler. Tanto el cálculo
   por separado como por lotes devuelve 5; los tiempos reescalados son 1000 y
   750. Esto comprueba integración y guidance, no el DiT real.
2. El latente de cámara `[0, 1, 0, 0.5, 0.25]` produce yaw 0, pitch 0, radio
   2 y campo de visión `degrees(2 * atan(0.125))`.
3. Una distribución octree con un único hijo de probabilidad 1 le asigna los
   siete elementos; el total se conserva independientemente del desplazamiento
   aleatorio.

## Ejecución

Prepara el entorno de referencia y una copia original en el commit indicado como
se describe en [configuración de entornos](environments.md). Desde la raíz:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -v
```

Ajusta `IDOL_REFERENCE_SOURCE` a la ubicación de tu copia. Comprueba el commit:

```sh
git -C ../idol-sdk-reference rev-parse HEAD
```

El resultado debe coincidir con el commit indicado. Si la variable no está
definida, las pruebas se marcan como omitidas; una ejecución omitida no es una
validación. Las pruebas no comprueban automáticamente la revisión de la copia; verifícala con el comando anterior. Las dependencias ausentes o una copia inutilizable pueden provocar errores.

## Siguientes pasos

Tras obtener acceso, compruébalo sin publicar tokens, registra la revisión del
modelo y ejecuta el pipeline original completo. Registra por separado la
configuración y artefactos que sean seguros para compartir; no incluyas
credenciales, imágenes privadas, rutas personales ni pesos en el repositorio.

Versión principal: [docs/reference-components.md](../docs/reference-components.md).
