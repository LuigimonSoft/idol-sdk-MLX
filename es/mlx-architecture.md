# Propuesta de arquitectura del backend MLX

Este documento define el diseño inicial del issue #9. Es una propuesta de
interfaces; la API aún no está implementada ni validada. El comportamiento
original y las dimensiones se describen en
[architecture.md](../docs/architecture.md). El
[SDK Idol original](https://github.com/speridlabs/idol-sdk) permanece como una
copia PyTorch separada para pruebas de equivalencia.

## Límites

- Implementar el backend como paquete separado `idol_mlx`. El runtime MLX no
  debe importar PyTorch, torchvision, Diffusers, Transformers ni TensorDict.
  Estos pueden permanecer en el entorno de referencia o en una herramienta de
  conversión explícitamente separada cuando sea necesaria.
- Usar arrays MLX dentro de los modelos y el muestreo. Pillow, NumPy y Python
  pueden utilizarse en los límites de imagen y exportación PLY. Evitar
  conversiones implícitas entre frameworks en cada paso.
- Conservar los conceptos públicos: imagen, modos synthetic/real, semilla,
  pasos, guidance, desplazamiento temporal, cantidad de Gaussianas, imagen
  preparada, cámara y exportación PLY. La compatibilidad debe validarse.
- La primera ruta completa admite una imagen. Las dimensiones de lote se
  conservan internamente, sin prometer todavía procesamiento por lotes.

## Organización propuesta

Estas rutas describen responsabilidades; se crearán al implementar cada issue.

| Módulo | Responsabilidad | Dependencias y límites |
| --- | --- | --- |
| `idol_mlx.pipeline` | Configuración, ciclo de vida, orquestación y operaciones de carga, preprocesamiento y generación. | Componentes MLX y tipos de entrada/salida; sin PyTorch. |
| `idol_mlx.config` | Validación de configuración, dimensiones y valores admitidos. | Python y datos tipados; sin acceso al dispositivo ni almacenamiento de secretos. |
| `idol_mlx.weights` | Localizar checkpoints convertidos y validar manifiesto, nombres, formas y tipos. | MLX y archivos locales; acceso al Hub separado de las definiciones. |
| `idol_mlx.preprocess` | Conversión de imagen, alfa, recorte, redimensionado y composición. | Pillow/NumPy e interfaz opcional de segmentación. |
| `idol_mlx.models.birefnet` | Máscara para imágenes sin alfa útil. | Capas MLX y pesos compatibles; la convolución deformable es un riesgo pendiente. |
| `idol_mlx.conditioning` y encoders | Características DINOv3 y VAE FLUX.2, con empaquetado y normalización. | MLX y checkpoints; sus accesos y licencias se revisan por separado. |
| `idol_mlx.models.layers` | Capas lineales, normalización, embeddings, atención y operaciones rotatorias. | Solo MLX; pruebas de equivalencia por componente. |
| `idol_mlx.models.dit` | Velocidades del latente y cámara según tiempo y condición de imagen. | Capas compartidas MLX y pesos del denoiser. |
| `idol_mlx.sampler` | Calendario Euler y classifier-free guidance con una función de velocidad. | Arrays MLX y entradas aleatorias explícitas. |
| `idol_mlx.models.decoder` | Probabilidades octree, muestreo de anclas y atributos gaussianos. | Capas MLX, pesos y aleatoriedad controlada. |
| `idol_mlx.camera` y `idol_mlx.gaussian` | Convertir resultados a cámara y representación gaussiana; exportar PLY. | MLX en el límite numérico y serialización en host al exportar. |
| `idol_mlx.sample` | Argumentos de CLI y selección de archivos de salida. | API pública; rutas elegidas por un usuario local confiable. |

La conversión de checkpoints es independiente: conserva la procedencia y haz
explícita la conversión. La carga debe fallar claramente ante tensores ausentes,
inesperados o incompatibles. Los mapeos y disposición de convoluciones se
definirán en el issue #14.

## Contratos de componentes

Las formas corresponden a la configuración conocida, no a cualquier checkpoint.
`B` es lote, `L` longitud común de tokens, `A` anclas y `G=32A`.
Los tipos y longitudes reales deben verificarse con el modelo entrenado.

| Límite | Entrada | Salida |
| --- | --- | --- |
| Preprocesamiento | Imagen o ruta local, modo y margen opcional; segmentación si se necesita. | Imagen RGB preparada y píxeles `[B, 3, 1024, 1024]` en [0, 1]. |
| Condición de imagen | Píxeles y valor/clave aleatoria controlada para el VAE. | `feature1 [B, L, 1280]` y `feature2 [B, L, 128]`; orden y preprocesamiento equivalentes al original. |
| DiT | Latente de objeto `[B, 8192, 16]`, cámara `[B, 1, 5]`, tiempo `[B]` y características. | Velocidades con las mismas formas que latente y cámara. |
| Sampler | Función de velocidad, ruido inicial, condiciones, calendario y guidance. | Latentes de objeto y cámara; una captura de depuración no debe divulgar entradas silenciosamente. |
| Octree | Latente, cantidad de anclas, nivel y aleatoriedad controlada. | Posiciones `[B, A, 3]`; cantidad y orden deben verificarse. |
| Decodificador gaussiano | Anclas y latente de objeto. | Valores `[B, A, 480]` y atributos transformados de `G` Gaussianas. |
| Cámara / exportación | Latente `[5]` y atributos gaussianos. | Yaw, pitch, radio, campo de visión y PLY con las convenciones originales. |

El original calcula `A=max(1, num_gaussians // 32)`. Los límites y redondeos
públicos deben documentarse después de probar el comportamiento; no se debe
asumir que cualquier cantidad solicitada se respeta exactamente.

## Esquema de API pública

Los nombres y firmas son provisionales hasta que las pruebas de integración
definan la compatibilidad:

```python
# Propuesta: este paquete y esta API todavía no existen.
from idol_mlx import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
result = pipeline("object.png", mode="real", seed=42, steps=50)
result.gaussian.save_ply("object.ply")
result.preprocessed_image.save("object_input.png")
# result.camera contiene los campos de la cámara.
```

La carga debe distinguir archivos ausentes, acceso restringido pendiente,
dispositivo no compatible, configuración inválida y checkpoint incompatible.
No debe cambiar silenciosamente a CPU o PyTorch mientras afirma usar Metal.
No se guardan ni registran tokens en la configuración; usa la autenticación del
Hub o un secret de entorno.

## Evaluación, precisión y aleatoriedad

MLX evalúa de forma diferida. Evalúa tensores en límites deliberados para
medir tiempos y memoria, serializar y comparar; no sincronices cada operación
sin medir el coste. Consulta
[evaluación diferida](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
y [memoria unificada](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

Define los tipos explícitamente por componente y conserva las normalizaciones
FP32 requeridas. El soporte FP16/BF16, la memoria y la equivalencia deben
medirse en el equipo. La compilación es una optimización posterior: tamaños
variables del octree, estado aleatorio, memoria y recompilaciones requieren
pruebas específicas.

Gestiona la aleatoriedad por generación. Usa claves PRNG MLX explícitas o valores
inyectables para el VAE, ruido inicial, octree y perturbación final. Una
referencia debe entregar los mismos tensores a ambos backends; una semilla igual
no garantiza secuencias iguales. Consulta
[claves aleatorias MLX](https://ml-explore.github.io/mlx/build/html/python/random.html).

## Orden de integración y validación

1. Empaquetado y detección de Metal (issues #10 y #12).
2. Carga estricta de configuración y pesos, y mapeo de conversión (#13–#14).
3. Capas compartidas y comparación con datos sintéticos (#17–#21), usando la
   [política provisional](../docs/parity-metrics.md).
4. DiT, sampler, cámara y decodificadores con ruido explícito (#22–#36).
5. Encoders y ruta inicial con alfa antes de BiRefNet (#37–#49), revisando
   licencias y accesos.
6. Validar pipeline completo, PLY y CLI; medir velocidad y memoria antes de
   optimizar o decidir si compilar (#50–#64).

Cada límite necesita pruebas de forma, tipo, errores y determinismo. La inferencia
y calidad completas necesitan pesos restringidos y quedan fuera de este diseño.
Un servicio o agente que acepte rutas no confiables necesita una barrera
separada; consulta [seguridad de rutas](output-path-security.md).

Versión principal: [docs/mlx-architecture.md](../docs/mlx-architecture.md).
