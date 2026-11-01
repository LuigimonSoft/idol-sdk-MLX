# Comparación de salidas entre backends

[English](../docs/parity-harness.md)

Esta herramienta preliminar compara tensores identificados por nombre en dos
archivos `safetensors` locales confiables, utilizando las
[métricas de paridad](parity-metrics.md). Genera la referencia y el resultado
candidato con las **mismas entradas y pesos capturados**; los valores aleatorios
generados por separado no son intercambiables. La herramienta solo compara
salidas: no ejecuta modelos, carga entradas en los backends, deduce disposiciones
ni demuestra que las entradas y pesos fueran iguales. Conserva esa trazabilidad
en la prueba correspondiente.

Usa el entorno de referencia de [entornos](environments.md) e indica las
tolerancias para cada componente:

```sh
.venv-reference/bin/python -m scripts.compare_parity reference.safetensors candidate.safetensors \
  --reference-key component.output --candidate-key output --atol 1e-5 --rtol 1e-4
```

El comando imprime un informe JSON con nombres, tolerancias, resultado,
valores finitos, número de elementos y fallos y errores absolutos máximo y
medio. Devuelve 0 si coincide, 1 si falla numéricamente (incluidos valores no
finitos) y 2 ante errores de archivo, esquema, forma o tolerancia. Un error
calculado infinito se representa como `"infinity"` para conservar JSON válido.
Las tolerancias son provisionales hasta calibrarlas para cada operación y equipo.

Las entradas deben ser tensores numéricos reales, no vacíos y de la misma
forma. La carga safetensors mediante NumPy no admite todos los tipos (en
particular BF16); exporta las salidas de comparación como FP32 después del
cálculo BF16 y registra esa conversión. Usa solo archivos locales confiables.
El comando no escribe archivos ni modifica capturas. No subas credenciales,
entradas privadas ni pesos restringidos.

Las pruebas cubren coincidencias, diferencias, errores y formato del informe.
La paridad PyTorch/MLX por componente aún requiere implementaciones MLX y
pruebas de integración con entradas compartidas; esta herramienta no la demuestra.
