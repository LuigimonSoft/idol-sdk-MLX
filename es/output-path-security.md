# Rutas locales de salida y límites de confianza

`scripts/capture_reference.py --output` y
`scripts/prepare_test_images.py --output` aceptan una ubicación elegida por
quien ejecuta el comando localmente. Esto es intencional: permite guardar los
artefactos fuera del repositorio en un directorio controlado por el usuario.
Quien invoca estos comandos debe considerar la ruta como confiable.

## Hallazgos S8707 de SonarCloud

SonarCloud reportó `pythonsecurity:S8707` para ambos comandos porque el
argumento de CLI llega a escrituras en el sistema de archivos. Actualmente son
utilidades locales de desarrollo; el repositorio no implementa un endpoint HTTP
ni un wrapper de agente para ellas. Sin embargo, los hallazgos identifican un
riesgo real **si** una integración pasa como ruta una solicitud, documento,
respuesta de modelo u otro valor controlado por un atacante.

El comportamiento actual no es una barrera de seguridad frente a quien controla
el destino. En particular, la creación exclusiva (`xb`) impide reemplazar un
archivo final existente, pero no limita el directorio de destino. El generador de
imágenes comprueba nombres existentes, incluidos enlaces simbólicos rotos, y
usa creación exclusiva; esto no protege completamente frente a cambios de
enlaces de directorio, carreras o un invocador malicioso.

## Barrera necesaria para integraciones futuras

No reenvíes una ruta no confiable a estos comandos ni a sus funciones de
escritura. Antes de conectarlos a un agente o servicio:

1. Define la raíz de salida mediante configuración confiable del operador,
   independiente de las solicitudes y del modelo; limita los permisos del proceso.
2. Acepta solo un identificador o nombre de archivo apropiado del invocador no
   confiable, rechaza rutas absolutas y traversal, y garantiza la contención con
   primitivas de sistema de archivos adecuadas a la plataforma.
3. Trata los enlaces simbólicos y las carreras entre validación y uso; comprobar
   un prefijo de texto o usar solo `resolve()` no basta frente a cambios
   concurrentes del sistema de archivos.
4. Añade pruebas de integración para traversal, rutas absolutas, enlaces
   simbólicos, carreras y sobrescritura. Vuelve a evaluar los hallazgos de
   SonarCloud con el flujo real resultante.

Los scripts siguen siendo utilidades locales confiables; no se han convertido
en servicios aislados. No marques los hallazgos como resueltos solo por usar
`xb` o una comprobación previa. Registra cualquier decisión de SonarCloud
según el despliegue real y revísala si cambia el límite de confianza.

Versión principal: [docs/output-path-security.md](../docs/output-path-security.md).
