# Decodificación del latente de cámara

[English](../docs/camera-decoding.md)

`idol_mlx.camera.decode_camera` convierte un latente de cámara MLX de forma
`(5,)` en los cuatro campos del [decodificador original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/camera.py):
`yaw` y `pitch` en radianes, `radius` y el campo de visión vertical `fov` en
grados. Devuelve un diccionario de valores `float` de Python, serializable a
JSON cuando los resultados son finitos. Los latentes por lote deben seleccionarse
individualmente; las formas incompatibles y entradas que no sean MLX producen
`ValueError`.

```python
import json
import mlx.core as mx
from idol_mlx.camera import decode_camera

camera = decode_camera(mx.array([0.0, 1.0, 0.0, 0.5, 0.25]))
camera_json = json.dumps(camera, allow_nan=False)
```

Como en la referencia, los tres primeros componentes definen la dirección; el
cuarto es la distancia inversa y el quinto el semiancho. Una norma de dirección
igual o inferior a `1e-6` usa `(0, 0, 1)`; una distancia inversa inferior a
`1e-3` usa radio 1000 como aproximación ortográfica. Desde ese umbral se usa
perspectiva. La implementación transfiere escalares a Python y no es una
operación diferenciable, por lotes ni completamente residente en GPU.

Las pruebas comparan latentes controlados FP32 y FP16 con la referencia PyTorch
en CPU, incluidos los dos modos de proyección y direcciones cercanas al umbral,
con `atol=2e-6`, `rtol=2e-6`. CI ejecuta las pruebas con MLX CPU y también se
ejecutaron localmente con Metal. Validan la decodificación de entradas
controladas, no la calidad de cámaras predichas ni la inferencia integral.
