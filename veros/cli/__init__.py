#!/usr/bin/env python

try:
    import click
    have_click = True
except ImportError:
    have_click = False

if not have_click:
    raise ImportError('The Veros command line tools require click (e.g. through `pip install click`)')

del click
del have_click

from . import veros, veros_run, veros_copy_setup, veros_create_mask, veros_resubmit

veros.cli.add_command(veros_run.cli, 'run')
veros.cli.add_command(veros_copy_setup.cli, 'copy-setup')
veros.cli.add_command(veros_create_mask.cli, 'create-mask')
veros.cli.add_command(veros_resubmit.cli, 'resubmit')
