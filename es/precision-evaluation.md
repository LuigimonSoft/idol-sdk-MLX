# Política de precisión y evaluación diferida

[English](../docs/precision-evaluation.md)

Esta política preliminar para futuros componentes no demuestra equivalencia del
modelo. `PrecisionPolicy()` selecciona float32 como tipo base de almacenamiento;
float16 y bfloat16 requieren una selección explícita y siguen siendo
experimentales hasta medir la precisión y compatibilidad de cada componente y
hardware. La política solo convierte arrays MLX de punto flotante al llamar a
`cast`. No convierte índices enteros ni máscaras, modifica ajustes globales,
altera la carga de checkpoints ni garantiza la precisión interna de las
operaciones. La conversión puede perder información.

MLX puede elegir algoritmos y precisión interna según el hardware incluso con
entradas float32. Registra versión de MLX, dispositivo, ajustes de precisión,
formas y tamaño del lote en las investigaciones de paridad. Consulta la
documentación oficial sobre [precisión numérica](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
y [tipos de datos](https://ml-explore.github.io/mlx/build/html/python/data_types.html).

Las operaciones MLX usan evaluación diferida. `cast` no materializa
explícitamente los resultados. Usa `evaluate_boundary` para llamar a `mx.eval`
en límites explícitos de componentes, pasos de muestreo o salida final, en vez
de hacerlo tras cada operación. Imprimir, convertir a NumPy, guardar, acceder a
un escalar y el flujo de control también pueden provocar evaluación implícita.
Evalúa antes de medir el tiempo y selecciona límites a partir de mediciones de
rendimiento y memoria. Consulta [evaluación diferida](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

```python
import mlx.core as mx
from idol_mlx.precision import PrecisionPolicy, evaluate_boundary

x = mx.array([1.0, 2.0])
x = PrecisionPolicy("float16").cast(x)  # selección experimental explícita
y = x * 2
evaluate_boundary("component", y)
```

Ejecuta la suite normal en el entorno de referencia. Para probar arrays MLX
reales en una Mac compatible, instala el paquete MLX y ejecuta:

```sh
.venv-mlx/bin/python -m unittest discover -s tests -p 'test_precision*.py' -v
```

La prueba con MLX real se omite si no está instalado. Estas comprobaciones no
garantizan paridad FP16/BF16 por operación ni del modelo completo; compara los
componentes reales con entradas compartidas y tolerancias calibradas antes de
adoptar precisión reducida.
