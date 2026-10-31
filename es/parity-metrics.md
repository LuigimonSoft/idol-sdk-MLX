# Métricas de equivalencia numérica

Este documento establece un método inicial para el issue #7. Los umbrales son
**puntos de partida provisionales**, no garantías medidas ni criterios finales
de aceptación. Calibra cada componente con referencias capturadas en el equipo
objetivo y registra los errores observados antes de adoptar un umbral. Un
resultado fuera de tolerancia requiere investigación; no identifica por sí solo
la causa.

## Comparación de tensores deterministas

Para una referencia PyTorch `r` y un resultado MLX `c`, compara formas y
entradas idénticas, utilizando valores capturados en vez de generar números
aleatorios por separado. Para cada elemento:

```text
abs(c - r) <= atol + rtol * abs(r)
```

Usa tolerancia absoluta cerca de cero y relativa para valores mayores. La
utilidad `scripts/parity_metrics.py` informa además el error absoluto máximo y
medio, el número total de elementos y los que quedan fuera de tolerancia.
Convierte tensores numéricos reales a FP64 para calcular, pero no recupera la
precisión perdida anteriormente.

Se rechazan formas distintas, arreglos vacíos, tolerancias inválidas y entradas
no numéricas o complejas. Cualquier NaN o infinito falla, aunque aparezca igual
en ambas entradas; en ese caso se informa el número de elementos no finitos y
no se calculan errores agregados. Valida estructura y finitud por separado de
la cercanía numérica.

## Política inicial por etapa

| Etapa | Comparación inicial | Estado de tolerancia / aceptación |
| --- | --- | --- |
| Disposición, formas, claves, orden de tokens, claves de cámara y cantidades | Comprobaciones estructurales exactas. | Exactas; sin tolerancia numérica. |
| Embeddings FP32, normalización, operaciones rotatorias, cámara y sampler de prueba | Comparación elemento a elemento y error máximo y medio. | Exploratoria: `atol=1e-5`, `rtol=1e-4`; calibrar por operación. Evaluar cada campo de cámara por separado, según sus unidades. |
| Atención FP16/BF16, bloques transformer y decodificadores | Salidas intermedias con precisión y entradas idénticas; registrar error por etapa. | Sin umbral aprobado. Medir en el equipo y elegir tolerancias por componente. |
| Trayectoria Euler | Comparar velocidades y estados por paso con ruido, condiciones y tiempos idénticos. | Sin umbral final; registrar el crecimiento del error en lugar de aplicar automáticamente la tolerancia FP32 al proceso completo. |
| Posterior del VAE y muestreo octree | Comparar primero cálculos deterministas compartiendo los valores aleatorios explícitos cuando ambas implementaciones lo permitan. | Sin criterio elemento a elemento hasta controlar la aleatoriedad; no comparar flujos independientes solo por tener la misma semilla. |
| Cantidades octree y salida gaussiana | Comprobar conservación de cantidades, finitud y atributos válidos; comparar valores correspondientes solo cuando aleatoriedad y orden estén controlados. | Invariantes exactas y tolerancias calibradas; cambios de orden o muestreo pueden necesitar pruebas de distribución. |
| Imagen a 3D completa | Comparar resultados y vistas renderizadas con entradas compartibles y configuración documentada. | Las métricas visuales y de distribución deben definirse y validarse; un único umbral elemento a elemento del PLY no basta. |

La captura sintética de RePo y atención permite compartir entradas y pesos al
principio; no es una salida del modelo entrenado. No declares validada una
comparación sin datos de referencia o ejecución MLX.

## Verificación

Con el entorno de referencia fijado, ejecuta desde la raíz:

```sh
.venv-reference/bin/python -m unittest discover -s tests -p 'test_parity_metrics.py' -v
```

Las pruebas cubren igualdad, tolerancia absoluta cerca de cero, tolerancia
relativa, métricas, valores no finitos, formas y entradas inválidas. La utilidad
evalúa arreglos suministrados por el usuario; la conversión desde PyTorch o MLX,
las políticas por componente y las referencias con pesos reales siguen siendo
trabajo independiente. No subas credenciales, entradas privadas ni pesos.

Versión principal: [docs/parity-metrics.md](../docs/parity-metrics.md).
