# Comparación reducida del denoiser con checkpoint entrenado

[English](../docs/dit-trained-validation.md)

Este es un diagnóstico del issue #24, **no una validación de paridad aprobada**.
Se descargaron temporalmente el checkpoint `denoiser.safetensors` y la
configuración de [`speridlabs/idol` revisión `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902).
El checkpoint contiene 560 tensores FP32 y se cargó estrictamente en PyTorch
original y MLX usando la configuración completa. No se subieron pesos ni datos
de entrada. La referencia se ejecutó explícitamente en FP32, no en su precisión
mixta predeterminada. Después ambos modelos usaron solo las primeras ocho
posiciones `pos_pe` y entradas sintéticas idénticas: lote 1, ocho tokens
latentes, cuatro tokens de condición y tiempo 500. Esta secuencia reducida no
valida la ejecución completa de 8192 tokens.

En un Apple M4 con 32 GB de memoria unificada, macOS 27.2, MLX 0.32.3 y
PyTorch 2.15.0, con tolerancias provisionales `atol=2e-5, rtol=2e-4`:

| Dispositivo MLX | Error absoluto máximo de posición | Error máximo latente | Error máximo cámara | Resultado |
| --- | ---: | ---: | ---: | --- |
| CPU | 0.0312334 | 0.0011998 | 0.0019016 | Fallo |
| GPU Metal | 0.0312334 | 0.0020876 | 0.0022813 | Fallo |

El embedding de posición difiere antes de los bloques del denoiser. Una
inspección separada encontró una diferencia absoluta máxima de frecuencias de
0.00390625 entre PyTorch CPU y MLX Metal, con diferencias en 56 de 170 valores.
Al proporcionar directamente a MLX las frecuencias calculadas por PyTorch, el
error máximo posicional bajó de 0.0312334 a 1.78814e-7 para esta entrada. Esto
aísla el cálculo de frecuencias como causa principal de esta diferencia
posicional. Una segunda ejecución MLX sustituye ese mismo embedding de
posición derivado de la referencia, sin cambiar el código de producción. Para
la misma entrada reducida y tolerancias, los resultados fueron:

| Dispositivo MLX | Error máximo posición | Error máximo latente | Error máximo cámara | Resultado con frecuencias compartidas |
| --- | ---: | ---: | ---: | --- |
| CPU | 1.19209e-7 | 7.33137e-6 | 1.49012e-5 | Pasa |
| GPU Metal | 1.78814e-7 | 5.12600e-6 | 1.10269e-5 | Pasa |

Estas mediciones adicionales usaron Python 3.12 en un entorno temporal de
diagnóstico (el paquete admite Python 3.13), con las mismas versiones de MLX
y PyTorch. Muestran el efecto para esta entrada FP32 reducida; no establecen
paridad general ni proporcionan una solución de producción sin PyTorch. La ruta
normal de MLX sigue fallando. Desactivar `MLX_ENABLE_TF32` no cambió los errores
de salida originales de GPU en este caso. Queda pendiente investigar y corregir las diferencias; no se han
ampliado las tolerancias para hacer pasar la prueba.

## Reproducir

Instala MLX y las dependencias de referencia fijadas en un mismo entorno. Obtén
los dos archivos bajo la licencia y condiciones aplicables; mantenlos locales.
Desde la raíz del proyecto, con el SDK original fijado en `PYTHONPATH`:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_dit_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/denoiser.safetensors --device gpu
```

Usa `--device cpu` para MLX CPU y `--tokens` o `--context-tokens` para elegir
longitudes reducidas. La salida JSON contiene solo métricas agregadas; el código
0 indica que las seis comparaciones pasan y el 1 que alguna falla. La herramienta
carga el modelo y checkpoint completos, por lo que necesita memoria suficiente.
No escribe pesos ni entradas. La prueba automatizada usa un checkpoint sintético
pequeño en CI; no requiere pesos entrenados. No valida memoria a tamaño completo,
encoders, decodificadores ni el resultado de imagen a 3D.
