BACKENDS = ('numpy', 'jax')

BACKEND_MESSAGES = {
    'jax': 'Kernels are compiled during first iteration, be patient'
}

_init_done = set()


def init_jax_config():
    if 'jax' in _init_done:
        return

    import jax
    from veros import runtime_settings
    from veros.state import (
        VerosState, VerosVariables, DistSafeVariableWrapper,
        veros_state_pytree_flatten, veros_state_pytree_unflatten,
        veros_variables_pytree_flatten, veros_variables_pytree_unflatten,
        dist_safe_wrapper_pytree_flatten, dist_safe_wrapper_pytree_unflatten
    )
    jax.config.update('jax_enable_x64', runtime_settings.float_type == 'float64')
    jax.config.update('jax_platform_name', runtime_settings.device)

    jax.tree_util.register_pytree_node(VerosState, veros_state_pytree_flatten, veros_state_pytree_unflatten)
    jax.tree_util.register_pytree_node(VerosVariables, veros_variables_pytree_flatten, veros_variables_pytree_unflatten)
    jax.tree_util.register_pytree_node(DistSafeVariableWrapper, dist_safe_wrapper_pytree_flatten, dist_safe_wrapper_pytree_unflatten)

    _init_done.add('jax')


def get_backend_module(backend_name):
    if backend_name not in BACKENDS:
        raise ValueError('unrecognized backend {} (must be either of: {!r})'
                         .format(backend_name, list(BACKENDS.keys())))

    backend_module = None

    if backend_name == 'jax':
        try:
            import jax  # noqa: F401
        except ImportError:
            pass
        else:
            init_jax_config()
            import jax.numpy as backend_module

    elif backend_name == 'numpy':
        import numpy as backend_module

    if backend_module is None:
        raise ValueError('backend "{}" failed to import'.format(backend_name))

    return backend_module


def get_curent_device_name():
    from veros import runtime_settings
    if runtime_settings.backend != 'jax':
        return 'cpu'

    return runtime_settings.device
