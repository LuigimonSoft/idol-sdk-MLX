# Comprobaciones de pull requests y SonarQube Cloud

El workflow de GitHub Actions ejecuta pruebas unitarias en pull requests, pushes
a `master` y ejecuciones manuales. Descarga el código original público en el
commit `4702a7f00850e34b6e83a2170cc123c3e9570744`, instala las dependencias
fijadas verificando hashes y define `IDOL_REFERENCE_SOURCE` para que las pruebas
de referencia se ejecuten en vez de omitirse.

El runner es macOS ARM64 de GitHub. Las pruebas usan CPU y no requieren
credenciales de Hugging Face ni pesos. No validan Metal, descargas restringidas
ni inferencia completa.

## Análisis automático de SonarQube Cloud

El análisis de SonarQube Cloud es **independiente del workflow de GitHub
Actions**. Con el proyecto vinculado a GitHub y el análisis automático activado,
SonarQube Cloud puede analizar los cambios de los PR y la rama predeterminada
mediante su integración con GitHub.

1. Importa o vincula `LuigimonSoft/idol-sdk-MLX` al proyecto global de
   SonarQube Cloud con organización `luigimonsoft` y clave
   `LuigimonSoft_idol-sdk-MLX`.
2. En **Administration > Analysis Method** del proyecto SonarQube Cloud, activa
   el análisis automático si el proyecto lo admite.
3. Abre o actualiza un PR y comprueba que el análisis aparezca en SonarQube Cloud
   y en GitHub. La protección de ramas y los checks obligatorios se configuran
   aparte; este cambio no los activa.

El análisis automático no importa la cobertura de las pruebas. El workflow no
invoca SonarScanner, no espera un quality gate ni entrega un token SonarCloud al
código del PR. Un job de pruebas aprobado no demuestra que SonarCloud haya
terminado correctamente: verifica ambos checks por separado.

Referencias: [configuración con GitHub](https://docs.sonarsource.com/sonarqube-cloud/getting-started/github),
[análisis automático](https://docs.sonarsource.com/sonarqube-cloud/discovering-sonarcloud/what-sonarcloud-can-do)
y [límites de cobertura](https://docs.sonarsource.com/sonarqube-cloud/analyzing-source-code/test-coverage/overview).

## Pruebas locales

Con el entorno de referencia y el checkout fijado descritos en
[environments.md](../docs/environments.md), ejecuta desde la raíz:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -v
```

No subas credenciales, imágenes privadas ni pesos. El workflow no necesita el
secret `SONAR_TOKEN` para el análisis automático.

Versión principal: [docs/ci-sonarcloud.md](../docs/ci-sonarcloud.md).
