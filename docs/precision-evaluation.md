# Precision and lazy evaluation policy

[Español](../es/precision-evaluation.md)

This is a preliminary policy for future components, not a claim of model
parity. `PrecisionPolicy()` chooses float32 as the baseline storage dtype;
float16 and bfloat16 require an explicit choice and remain experimental until
component-specific accuracy and hardware support are measured. A policy only
casts floating-point MLX arrays when its `cast` method is called. It does not
cast integer indices or masks, mutate global settings, alter checkpoint loading,
or guarantee the internal precision of an operation. Casting loses information
when the selected dtype cannot represent the original value.

MLX may choose hardware-dependent algorithms and internal precision even for
float32 inputs. Record MLX version, device, precision settings, shapes and
batch size in parity investigations; see the official
[numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
and [data types](https://ml-explore.github.io/mlx/build/html/python/data_types.html)
documentation. Do not assume reduced-precision storage is validated for a model.

MLX operations are lazy. `cast` does not explicitly materialize results.
Use `evaluate_boundary` to call `mx.eval` at an explicit component result,
sampling step, or final output boundary, not after every individual operation.
Printing, conversion to NumPy, saving, scalar access and control flow can also
trigger implicit evaluation. Evaluate before measuring elapsed execution time;
select boundaries based on actual performance and memory measurements. See
[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

```python
import mlx.core as mx
from idol_mlx.precision import PrecisionPolicy, evaluate_boundary

x = mx.array([1.0, 2.0])
x = PrecisionPolicy("float16").cast(x)  # explicit experimental selection
y = x * 2
evaluate_boundary("component", y)
```

Run the standard unit suite in the reference environment. To additionally
exercise real MLX arrays on a compatible Mac, install the MLX package and run:

```sh
.venv-mlx/bin/python -m unittest discover -s tests -p 'test_precision*.py' -v
```

The real-MLX test is skipped when MLX is absent. These checks do not establish
operation-level or end-to-end FP16/BF16 parity; compare actual components with
shared inputs and calibrated tolerances before adopting reduced precision.
