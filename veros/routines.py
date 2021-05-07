import functools
import inspect
import threading
from contextlib import ExitStack, contextmanager, nullcontext

from veros import logger

from veros.state import VerosState


# stack helpers

class RoutineStack:
    def __init__(self):
        self.keep_full_stack = False
        self._stack = []
        self._current_idx = []

    @property
    def stack_level(self):
        return len(self._current_idx)

    def append(self, val):
        frame = self._stack
        for i in self._current_idx:
            frame = frame[i][1]

        self._current_idx.append(len(frame))
        frame.append([val, []])

    def pop(self):
        frame = self._stack
        for i in self._current_idx[:-1]:
            frame = frame[i][1]

        if self.keep_full_stack:
            last_val = frame[-1][0]
        else:
            last_val = frame.pop()[0]
        self._current_idx.pop()
        return last_val


# global context

CURRENT_CONTEXT = threading.local()
CURRENT_CONTEXT.is_dist_safe = True
CURRENT_CONTEXT.routine_stack = RoutineStack()


@contextmanager
def record_routine_stack():
    stack = CURRENT_CONTEXT.routine_stack
    if not stack.keep_full_stack:
        stack.keep_full_stack = True
        reset = True

    try:
        yield stack._stack
    finally:
        if reset:
            stack.keep_full_stack = False


@contextmanager
def enter_routine(name, routine_obj, timer=None, dist_safe=True):
    from veros import runtime_state as rst
    from veros.distributed import abort
    stack = CURRENT_CONTEXT.routine_stack

    logger.trace('{}> {}', '-' * stack.stack_level, name)
    stack.append(routine_obj)

    reset_dist_safe = False
    if CURRENT_CONTEXT.is_dist_safe:
        if not dist_safe and rst.proc_num > 1:
            CURRENT_CONTEXT.is_dist_safe = False
            reset_dist_safe = True

    timer_ctx = nullcontext() if timer is None else timer

    try:
        with timer_ctx:
            yield

    except:  # noqa: E722
        if reset_dist_safe:
            abort()
        raise

    finally:
        if reset_dist_safe:
            CURRENT_CONTEXT.is_dist_safe = True

        r = stack.pop()
        assert r is routine_obj

        exec_time = ''
        if timer is not None:
            exec_time = f'({timer.get_last_time():.3f}s)'

        logger.trace('<{} {} {}', '-' * stack.stack_level, name, exec_time)


# helper functions

def _get_func_name(function):
    return f'{inspect.getmodule(function).__name__}:{function.__qualname__}'


def _is_method(function):
    if inspect.ismethod(function):
        return True

    # hack for unbound methods: check if first argument is called "self"
    spec = inspect.getfullargspec(function)
    return spec.args and spec.args[0] == 'self'


# routine

def veros_routine(function=None, *, dist_safe=True, local_variables=()):
    """
    .. note::

      This decorator should be applied to all functions that access the Veros state object
      (even when subclassing :class:`veros.VerosSetup`).

    The first argument to the decorated function must be a VerosState instance.

    Veros routines cannot return anything. All changes must be applied to the passed state object.

    Parameters:
        dist_safe (bool): If set to False, all variables specified in local_variables are synced
            to the root process before execution and synced back after. This means that the routine
            will only be executed on rank 0. Has no effect in non-distributed contexts.

        local_variables (Tuple[str]): List of variable names to be synced if dist_safe=False. This
            must include all variables retrieved from the state object throughout the routine (inputs
            *and* outputs).

    Example:
       >>> from veros import VerosSetup, veros_routine
       >>>
       >>> class MyModel(VerosSetup):
       >>>     @veros_routine
       >>>     def set_topography(self, state):
       >>>         vs = state.variables
       >>>         settings = state.settings
       >>>         vs.kbot = np.random.randint(0, settings.nz, size=vs.kbot.shape)

    """
    def inner_decorator(function):
        narg = 1 if _is_method(function) else 0
        num_params = len(inspect.signature(function).parameters)
        if narg >= num_params:
            raise TypeError('Veros routines must take at least one argument')

        routine = VerosRoutine(function, state_argnum=narg, dist_safe=dist_safe, local_variables=local_variables)
        routine = functools.wraps(function)(routine)
        return routine

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class VerosRoutine:
    """Do not instantiate directly!"""

    def __init__(self, function, dist_safe=True, local_variables=(), state_argnum=0):
        if isinstance(local_variables, str):
            local_variables = (local_variables,)

        self.function = function
        self.dist_safe = dist_safe
        self.local_variables = local_variables
        self.state_argnum = state_argnum
        self.name = _get_func_name(self.function)

        self._traced = False
        self.inputs = None
        self.outputs = None

    def __call__(self, *args, **kwargs):
        from veros import runtime_state as rst
        from veros.state import VerosState, DistSafeVariableWrapper
        from veros.core.operators import flush

        veros_state = args[self.state_argnum]

        if not isinstance(veros_state, VerosState):
            raise TypeError(f'Argument {self.state_argnum} to this Veros routine must be a VerosState object')

        timer = veros_state.profile_timers[self.name]

        with ExitStack() as es:
            vars_initialized = veros_state._variables is not None

            if vars_initialized:
                es.enter_context(veros_state.variables.unlock())

            if vars_initialized and not self._traced:
                inputs = {}
                outputs = {}

                inputs['var'], outputs['var'] = es.enter_context(veros_state.variables.trace())
                inputs['settings'], outputs['settings'] = es.enter_context(veros_state.settings.trace())

            execute = True
            restore_vars = False

            if not self.dist_safe:
                orig_vars = veros_state._variables
                if not isinstance(orig_vars, DistSafeVariableWrapper):
                    restore_vars = True
                    veros_state._variables = DistSafeVariableWrapper(orig_vars, self.local_variables)
                    veros_state._variables._gather_variables()

                execute = rst.proc_rank == 0

            routine_ctx = enter_routine(
                name=self.name, routine_obj=self, timer=timer, dist_safe=self.dist_safe
            )

            out = None
            try:
                with routine_ctx:
                    if execute:
                        out = self.function(*args, **kwargs)

            finally:
                if restore_vars:
                    veros_state._variables._scatter_variables()
                    veros_state._variables = orig_vars

                flush()

        if out is not None:
            logger.warning(f"Routine {self.name} returned object of type {type(out)}. Return objects are silently dropped.")

        if vars_initialized and not self._traced:
            self.inputs = inputs
            self.outputs = outputs
            self._traced = True

    def __get__(self, instance, _):
        return functools.partial(self.__call__, instance)

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name} at {hex(id(self))}>'


# kernel

def veros_kernel(function=None, *, static_args=()):
    """Decorator that marks a function as a kernel that can be JIT compiled if supported
    by the backend.

    Kernels cannot modify the Veros state object. Instead, all modifications have to be
    returned explicitly.

    Parameters:
        static_args (Tuple[str]): Names of kernel arguments that should be static.

    Example:
        >>> from veros import veros_kernel, KernelOutput
        >>>
        >>> @veros_kernel
        >>> def double_psi(state):
        >>>     vs = state.variables
        >>>     vs.psi = 2 * vs.psi
        >>>     return KernelOutput(psi=vs.psi)

    """
    def inner_decorator(function):
        kernel = VerosKernel(function, static_args=static_args)
        kernel = functools.wraps(function)(kernel)
        return kernel

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class VerosKernel:
    """Do not instantiate directly!"""

    def __init__(self, function, static_args=()):
        """Do some parameter introspection."""
        # make sure function signature is in the form we need
        self.name = _get_func_name(function)
        self.func_sig = inspect.signature(function)

        func_params = self.func_sig.parameters

        allowed_param_types = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        if any(p.kind not in allowed_param_types for p in func_params.values()):
            raise ValueError(f'Veros kernels do not support *args, **kwargs, or keyword-only parameters ({self.name})')

        # parse static args
        if isinstance(static_args, str):
            static_args = (static_args,)

        func_argnames = list(func_params.keys())
        self.static_argnums = []
        for static_arg in static_args:
            try:
                arg_index = func_argnames.index(static_arg)
            except ValueError:
                raise ValueError(
                    f'Veros kernel {self.name} has no argument "{static_arg}", but it is given in static_args'
                ) from None

            self.static_argnums.append(arg_index)

        self.function = function

        self._traced = False
        self.inputs = None
        self.outputs = None

    def __call__(self, *args, **kwargs):
        from veros import runtime_settings
        from veros.core.operators import flush

        # apply JIT
        if runtime_settings.backend == "jax":
            import jax
            from jaxlib.xla_extension.jax_jit import CompiledFunction
            if not isinstance(self.function, CompiledFunction):
                self.function = jax.jit(self.function, static_argnums=self.static_argnums)

        # JAX only accepts positional args when using static_argnums
        # so convert everything to positional for consistency
        bound_args = self.func_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        veros_state = None
        for argval in bound_args.arguments.values():
            if isinstance(argval, VerosState):
                veros_state = argval
                break

        called_with_state = veros_state is not None

        # when profiling, make sure all inputs are ready before starting the timer
        if runtime_settings.profile_mode:
            flush()

        if called_with_state:
            timer = veros_state.profile_timers[self.name]
        else:
            timer = None

        with ExitStack() as es:
            if called_with_state:
                var, settings = veros_state.variables, veros_state.settings
                es.enter_context(var.unlock())

                if self._traced:
                    # hack to ensure that tracing callbacks are executed
                    # for already traced functions
                    for v in self.inputs['var']:
                        getattr(var, v)

                    for s in self.inputs['settings']:
                        getattr(settings, s)
                else:
                    inputs = {}
                    inputs['var'], _ = es.enter_context(var.trace())
                    inputs['settings'], _ = es.enter_context(settings.trace())
            else:
                # no state object -> no traceable inputs
                inputs = dict(var=set(), settings=set())

            with enter_routine(self.name, self, timer):
                out = self.function(*bound_args.arguments.values())

                if runtime_settings.profile_mode:
                    flush()

        if not self._traced:
            self.inputs = inputs
            self.outputs = dict(var=set(), settings=set())
            self._traced = True

        return out

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name} at {hex(id(self))}>'


def is_veros_routine(func):
    if isinstance(func, functools.partial):
        func = func.func

    if inspect.ismethod(func):
        func = func.__self__

    return isinstance(func, VerosRoutine)
