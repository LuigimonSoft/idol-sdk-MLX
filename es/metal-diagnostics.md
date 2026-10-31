# Diagnóstico de MLX y Metal

`idol_mlx.device.require_metal()` comprueba el entorno macOS con Apple Silicon
y ejecuta una pequeña operación MLX evaluada explícitamente en la GPU. Si falla
la plataforma, la inicialización de MLX, la disponibilidad de Metal o la
operación en GPU, lanza `MetalUnavailableError` con un mensaje de diagnóstico.

La comprobación no cambia el dispositivo predeterminado de MLX. Un resultado
correcto confirma solo esa operación; no demuestra que el modelo quepa en
memoria, que funcione la inferencia ni que coincida con PyTorch.

## Ejecución

Instala el paquete como se describe en
[mlx-package.md](../docs/mlx-package.md) y ejecuta:

```sh
.venv-mlx/bin/python -c "from idol_mlx.device import require_metal; require_metal(); print('Metal check passed')"
```

Si el proceso no puede acceder a una GPU, prueba en una terminal local normal
sobre un Mac con Apple Silicon. Una sesión restringida o sin interfaz gráfica
puede impedir el acceso a Metal aun cuando el equipo tenga una GPU compatible.
Si falta MLX o no puede inicializarse, comprueba el entorno Python activo y
reinstala el paquete. En otras plataformas, la comprobación rechaza la ejecución;
no cambia silenciosamente a CPU o PyTorch.

## Verificación

Las pruebas unitarias simulan plataformas admitidas y no admitidas, ausencia de
MLX o sus dependencias, errores de inicialización, Metal no disponible y fallo
de evaluación en GPU. En el Mac Apple Silicon de referencia, la comprobación real
pasó sin cambiar el dispositivo predeterminado y también con CPU seleccionado
como predeterminado. No se necesitan pesos, credenciales ni datos privados.

Versión principal: [docs/metal-diagnostics.md](../docs/metal-diagnostics.md).
