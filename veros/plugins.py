from collections import namedtuple

from veros.variables import Variable
from veros.settings import Setting

VerosPlugin = namedtuple('VerosPlugin', [
    'name',
    'module',
    'setup_entrypoint',
    'run_entrypoint',
    'settings',
    'variables',
    'conditional_variables',
    'diagnostics',
])


def load_plugin(module):
    from veros.diagnostics.base import VerosDiagnostic

    if not hasattr(module, '__VEROS_INTERFACE__'):
        raise RuntimeError('module {} is not a valid Veros plugin'.format(module.__name__))

    interface = module.__VEROS_INTERFACE__

    setup_entrypoint = interface.get('setup_entrypoint')

    if not callable(setup_entrypoint):
        raise RuntimeError('module {} is missing a valid setup entrypoint'.format(module.__name__))

    run_entrypoint = interface.get('run_entrypoint')

    if not callable(run_entrypoint):
        raise RuntimeError('module {} is missing a valid run entrypoint'.format(module.__name__))

    name = interface.get('name', module.__name__)

    settings = interface.get('settings', [])
    for setting, val in settings.items():
        if not isinstance(val, Setting):
            raise TypeError('got unexpected type {} for setting {}'.format(type(val), setting))

    variables = interface.get('variables', [])
    for variable, val in variables.items():
        if not isinstance(val, Variable):
            raise TypeError('got unexpected type {} for variable {}'.format(type(val), variable))

    conditional_variables = interface.get('conditional_variables', [])
    for _, sub_variables in conditional_variables.items():
        for variable, val in sub_variables.items():
            if not isinstance(val, Variable):
                raise TypeError('got unexpected type {} for variable {}'.format(type(val), variable))

    diagnostics = interface.get('diagnostics', [])
    for diagnostic in diagnostics:
        if not issubclass(diagnostic, VerosDiagnostic):
            raise TypeError('got unexpected type {} for diagnostic {}'.format(type(diagnostic), diagnostic))

    return VerosPlugin(
        name=name,
        module=module,
        setup_entrypoint=setup_entrypoint,
        run_entrypoint=run_entrypoint,
        settings=settings,
        variables=variables,
        conditional_variables=conditional_variables,
        diagnostics=diagnostics
    )
