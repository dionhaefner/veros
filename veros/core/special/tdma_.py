try:
    from . import tdma_cython_
except ImportError:
    HAS_CPU_EXT = False
else:
    HAS_CPU_EXT = True

try:
    from . import tdma_cuda_
except ImportError:
    HAS_GPU_EXT = False
else:
    HAS_GPU_EXT = True

import numpy as np
import jax.numpy as jnp

import jax
from jax import abstract_arrays
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla

_ops = xla_client.ops

if HAS_CPU_EXT:
    for kernel_name in (b'tdma_cython_double', b'tdma_cython_float'):
        fn = tdma_cython_.cpu_custom_call_targets[kernel_name]
        xla_client.register_custom_call_target(kernel_name, fn, platform="cpu")

if HAS_GPU_EXT:
    for kernel_name in (b'tdma_cuda_double', b'tdma_cuda_float'):
        fn = tdma_cuda_.gpu_custom_call_targets[kernel_name]
        xla_client.register_custom_call_target(kernel_name, fn, platform="gpu")


def _constant_s64_scalar(c, x):
    return _ops.Constant(c, np.int64(x))


def tdma(a, b, c, d, interior_mask, edge_mask, device=None):
    if device is None:
        device = jax.default_backend()

    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError('all inputs must have identical shape')

    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError('all inputs must have the same dtype')

    if device == "cpu":
        system_depths = jnp.sum(interior_mask, axis=-1, dtype='int32')
        return tdma_p.bind(a, b, c, d, system_depths)

    a = interior_mask * a * jnp.logical_not(edge_mask)
    b = jnp.where(interior_mask, b, 1.)
    c = interior_mask * c
    d = interior_mask * d

    return tdma_p.bind(a, b, c, d, system_depths=None)


def tdma_impl(*args, **kwargs):
    return xla.apply_primitive(tdma_p, *args, **kwargs)


def tdma_xla_encode_cpu(builder, a, b, c, d, system_depths):
    if not HAS_CPU_EXT:
        raise RuntimeError("CPU TDMA extensions could not be imported")

    x_shape = builder.GetShape(a)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if dtype not in supported_dtypes:
        raise TypeError('TDMA only supports {} arrays, got: {}'.format(supported_dtypes, dtype))

    # compute number of elements to vectorize over
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el

    stride = dims[-1]

    sys_depth_shape = builder.get_shape(system_depths)
    sys_depth_dtype = sys_depth_shape.element_type()
    sys_depth_dims = sys_depth_shape.dimensions()
    assert sys_depth_dtype is np.dtype(np.int32)
    assert tuple(sys_depth_dims) == tuple(dims[:-1])

    arr_shape = xla_client.Shape.array_shape(dtype, dims)
    out_shape = xla_client.Shape.tuple_shape([
        arr_shape,
        xla_client.Shape.array_shape(dtype, (stride,))
    ])

    if dtype is np.dtype(np.float32):
        kernel = b'tdma_cython_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'tdma_cython_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    out = _ops.CustomCall(
        builder,
        kernel,
        operands=(
            a, b, c, d,
            system_depths,
            _constant_s64_scalar(builder, num_systems),
            _constant_s64_scalar(builder, stride),
        ),
        shape=out_shape,
    )
    return _ops.GetTupleElement(out, 0)


def tdma_xla_encode_gpu(builder, a, b, c, d, system_depths):
    if not HAS_GPU_EXT:
        raise RuntimeError("GPU TDMA extensions could not be imported")

    if system_depths is not None:
        raise ValueError("TDMA does not support system_depths argument on GPU")

    a_shape = builder.get_shape(a)
    dtype = a_shape.element_type()
    dims = a_shape.dimensions()

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if dtype not in supported_dtypes:
        raise TypeError('TDMA only supports {} arrays, got: {}'.format(supported_dtypes, dtype))

    # compute number of elements to vectorize over
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el

    system_depth = dims[-1]

    if dtype is np.dtype(np.float32):
        kernel = b'tdma_cuda_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'tdma_cuda_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    opaque = tdma_cuda_.build_tridiag_descriptor(num_systems, system_depth)

    ndims = len(dims)
    arr_layout = tuple(range(ndims - 2, -1, -1)) + (ndims - 1,)
    arr_shape = xla_client.Shape.array_shape(dtype, dims, arr_layout)
    out_shape = xla_client.Shape.tuple_shape([arr_shape, arr_shape])

    out = _ops.CustomCallWithLayout(
        builder,
        kernel,
        operands=(a, b, c, d),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(arr_shape,) * 4,
        opaque=opaque
    )
    return _ops.GetTupleElement(out, 0)


def tdma_abstract_eval(a, b, c, d, system_depths):
    return abstract_arrays.ShapedArray(a.shape, a.dtype)


tdma_p = Primitive('tdma')
tdma_p.def_impl(tdma_impl)
tdma_p.def_abstract_eval(tdma_abstract_eval)

xla.backend_specific_translations['cpu'][tdma_p] = tdma_xla_encode_cpu
xla.backend_specific_translations['gpu'][tdma_p] = tdma_xla_encode_gpu
