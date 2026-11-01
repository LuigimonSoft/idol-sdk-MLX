# Sampler Euler preliminar

[English](../docs/euler-sampler.md)

`idol_mlx.sampler.sample_euler` implementa la integración Euler condicional del [sampler original fijado](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/sampler.py).
`time_grid` crea una secuencia descendente de 1 a 0 y aplica la transformación
temporal original. En cada paso el modelo recibe la muestra actual, un vector
de tiempos FP32 multiplicados por 1000 y la condición suministrada. El estado
siguiente resta la velocidad multiplicada por el intervalo temporal. Quien llama
proporciona el ruido inicial; el sampler no genera valores aleatorios y mantiene
la evaluación diferida de MLX.

La guía classifier-free opcional usa `(1 + w) * v_cond - w * v_neg` cuando
`cfg_strength > 0` y requiere `neg_condition`. El valor predeterminado es cero
(sin guía); valores cero o negativos omiten la pasada negativa, como en el
original. `batch_cfg=True` concatena entradas condicionales y negativas en una
sola llamada por paso. Solo equivale a llamadas separadas cuando el modelo trata
los elementos del lote independientemente. Quien llama debe proporcionar
modelo y tensores compatibles. `steps` debe ser un entero
positivo y `rescale_t` finito y positivo. La forma equivalente del denominador
`(1-t) + rescale_t*t` evita cancelación cerca del inicio. Para valores muy
pequeños preserva el extremo matemático aunque la expresión original pierda
precisión numérica.

Las pruebas sintéticas FP32 comparan el calendario y las salidas completas con
PyTorch original usando velocidades constantes y dependientes del tiempo y la
muestra, varios pasos, desplazamientos y un lote de dos. Las pruebas de guía
cubren ambos modos e intensidades positivas y no positivas con un modelo
sintético independiente por lote. Las tolerancias
provisionales para salidas son `atol=1e-5, rtol=1e-4`. No validan modelos entrenados, comportamiento con elementos del lote acoplados,
FP16/BF16, todos los rangos numéricos ni inferencia completa. Falta medir la
memoria de ambos modos con un modelo real.

Con MLX, PyTorch y el SDK original fijado instalados:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_sampler_mlx.py' -v
```

Usa `IDOL_TEST_MLX_DEVICE=gpu` para MLX en Metal cuando esté disponible;
PyTorch permanece en CPU. CI ejecuta estas pruebas de paridad con MLX CPU.
