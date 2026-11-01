# Exportación PLY de gaussianas

[English](../docs/gaussian-ply.md)

`idol_mlx.ply.save_gaussian_ply` exporta un elemento del lote desde los arrays
FP32 activados de la [conversión de atributos gaussianos](gaussian-attributes.md):

```python
from idol_mlx.ply import save_gaussian_ply

# attributes es el diccionario por lotes de decode_gaussian_attributes.
save_gaussian_ply("/trusted/output/object.ply", attributes, batch_index=0)
```

La ruta la selecciona un llamador local confiable y debe terminar en `.ply`.
La función crea un archivo nuevo de forma exclusiva; nunca reemplaza uno
existente. Valida las formas y los valores finitos de entrada y transformados
antes de abrir el archivo. Solo exporta el elemento de lote indicado (por
defecto 0); exporta los demás por separado. Un fallo de E/S después de abrir el
archivo puede dejarlo incompleto. Una colección vacía produce un PLY sin vértices.

El PLY binario little-endian contiene las propiedades originales en orden:
`x`, `y`, `z`, `nx`, `ny`, `nz`, `f_dc_0`–`f_dc_2`, `opacity`,
`scale_0`–`scale_2` y `rot_0`–`rot_3`. Las posiciones y cuaterniones se rotan
del eje z al eje y vertical; las normales son cero y el color conserva los
coeficientes de armónicos esféricos de grado cero. La opacidad se limita a
`[1e-6, 1 - 1e-6]` y convierte a logits; las escalas se limitan a `1e-6` y
convierten a logaritmos. Los cuaterniones usan el orden `(w, x, y, z)`. Estas
convenciones coinciden con el exportador original fijado; no se aplica una
conversión a RGB ni una normalización adicional de cuaterniones.

Las pruebas automáticas comparan cada campo con el exportador PyTorch original
para datos sintéticos en CPU y Metal, incluido el flujo de atributos a PLY,
rotación de ejes, límites, archivo vacío y creación exclusiva. Otra prueba
decodifica valores binarios conocidos con la biblioteca estándar de Python,
independientemente del lector PLY. También se abrió
y renderizó localmente una esfera coloreada sintética de 1500 gaussianas en el
[visor antimatter15/splat](https://github.com/antimatter15/splat/tree/ba182b51b7c2ad5738cdd6741cd63336d27470fb),
commit `ba182b51b7c2ad5738cdd6741cd63336d27470fb`, servido en localhost en el
navegador integrado. El visor leyó directamente el PLY exportado; no se subieron
datos del usuario ni pesos. Esto valida un visor y una entrada sintética, no
todos los visores ni la inferencia integral de imagen a 3D. Los límites de
cantidad y memoria de gaussianas se abordan por separado en el issue #36.
