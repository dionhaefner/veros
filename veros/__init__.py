"""Veros, the versatile ocean simulator"""

import sys
import types

# black magic: ensure lazy imports for public API by overriding module.__class__

def _reraise_exceptions(func):
    import functools
    raised = False

    @functools.wraps(func)
    def reraise_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ImportError('Critical error during initial import') from e

    return reraise_wrapper


class _PublicAPI(types.ModuleType):
    @property
    @_reraise_exceptions
    def __version__(self):
        from veros._version import get_versions
        return get_versions()['version']

    @property
    @_reraise_exceptions
    def logger(self):
        if not hasattr(self, '_logger'):
            from veros.logs import setup_logging
            self._logger = setup_logging()
        return self._logger

    @property
    @_reraise_exceptions
    def runtime_settings(self):
        if not hasattr(self, '_runtime_settings'):
            from veros.runtime import RuntimeSettings
            self._runtime_settings = RuntimeSettings()
        return self._runtime_settings

    @property
    @_reraise_exceptions
    def runtime_state(self):
        if not hasattr(self, '_runtime_state'):
            from veros.runtime import RuntimeState
            self._runtime_state = RuntimeState()
        return self._runtime_state

    @property
    @_reraise_exceptions
    def veros_routine(self):
        from veros.routines import veros_routine
        return veros_routine

    @property
    @_reraise_exceptions
    def veros_kernel(self):
        from veros.routines import veros_kernel
        return veros_kernel

    @property
    @_reraise_exceptions
    def KernelOutput(self):
        from veros.state import KernelOutput
        return KernelOutput

    @property
    @_reraise_exceptions
    def VerosSetup(self):
        from veros.veros import VerosSetup
        return VerosSetup

    @property
    @_reraise_exceptions
    def VerosState(self):
        from veros.state import VerosState
        return VerosState

    @property
    @_reraise_exceptions
    def VerosLegacy(self):
        from veros.veros_legacy import VerosLegacy
        return VerosLegacy

sys.modules[__name__].__class__ = _PublicAPI

del sys
del types
del _PublicAPI
del _reraise_exceptions
