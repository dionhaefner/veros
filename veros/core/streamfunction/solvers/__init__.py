import functools

from veros import runtime_settings as rs, runtime_state as rst, logger


def memoize(func):
    cache = {}
    @functools.wraps(func)
    def inner(*args):
        if args not in cache:
            cache[args] = func(*args)

        return cache[args]

    return inner


def _get_solver_class():
    ls = rs.linear_solver

    def _get_best_solver():
        if rst.proc_num > 1:
            try:
                from .petsc_ import PETScSolver
            except ImportError:
                logger.warning('PETSc linear solver not available, falling back to SciPy')
            else:
                return PETScSolver

        from .scipy import SciPySolver
        return SciPySolver

    if ls == 'best':
        return _get_best_solver()
    elif ls == 'petsc':
        from .petsc_ import PETScSolver
        return PETScSolver
    elif ls == 'scipy':
        from .scipy import SciPySolver
        return SciPySolver

    raise ValueError('unrecognized linear solver %s' % ls)


@memoize
def get_linear_solver(state):
    SolverClass = _get_solver_class()
    return SolverClass(state)
