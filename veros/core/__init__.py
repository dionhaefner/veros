import os
import importlib

from veros import logger


def build_all():
    """Trigger first import of all core modules"""
    from veros import runtime_settings as rs
    from veros.backend import BACKEND_MESSAGES, get_curent_device_name

    logger.info('Importing core modules')

    logger.opt(colors=True).info(
        ' Using computational backend <bold>{}</bold> on <bold>{}</bold>',
        rs.backend, get_curent_device_name()
    )
    extra_message = BACKEND_MESSAGES.get(rs.backend)
    if extra_message:
        logger.info('  {}', extra_message)

    basedir = os.path.dirname(__file__)

    for root, dirs, files in os.walk(basedir):
        py_path = '.'.join(os.path.split(os.path.relpath(root, basedir))).strip('.')

        for f in files:
            modname, ext = os.path.splitext(f)
            if modname.endswith('_') or ext != '.py':
                continue

            if py_path:
                module_path = f'veros.core.{py_path}.{modname}'
            else:
                module_path = f'veros.core.{modname}'

            logger.trace('importing {}', module_path)
            try:
                importlib.import_module(module_path)
            except ImportError:
                pass

    if not rs.__locked__:
        rs.__locked__ = True
        logger.info(' Runtime settings are now locked')

    logger.info('')

build_all()
