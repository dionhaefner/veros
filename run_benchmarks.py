#! /usr/bin/env python

import sys
import os
import subprocess
import multiprocessing
import importlib
import importlib.util
import re
import time
import math
import itertools

import click
import numpy as np
import ruamel.yaml as yaml

"""
Runs selected Veros benchmarks back to back and writes timing results to a YAML file.
"""

TESTDIR = os.path.join(os.path.dirname(__file__), os.path.relpath('benchmarks'))
COMPONENTS = ['numpy', 'numpy-mpi', 'jax', 'jax-gpu', 'jax-mpi', 'fortran', 'fortran-mpi']
STATIC_SETTINGS = '-v debug -s nx {nx} -s ny {ny} -s nz {nz} -s default_float_type {float_type} --timesteps {timesteps}'
BENCHMARK_COMMANDS = {
    'numpy': '{python} {filename} -b numpy ' + STATIC_SETTINGS,
    'numpy-mpi': '{mpiexec} -n {nproc} {python} {filename} -b numpy -n {decomp} ' + STATIC_SETTINGS,
    'bohrium': 'OMP_NUM_THREADS={nproc} BH_STACK=openmp BH_OPENMP_PROF=1 {python} {filename} -b bohrium '  + STATIC_SETTINGS,
    'bohrium-opencl': 'BH_STACK=opencl BH_OPENCL_PROF=1 {python} {filename} -b bohrium ' + STATIC_SETTINGS,
    'bohrium-cuda': 'BH_STACK=cuda BH_CUDA_PROF=1 {python} {filename} -b bohrium ' + STATIC_SETTINGS,
    'bohrium-mpi': 'OMP_NUM_THREADS=1 {mpiexec} -n {nproc} {python} {filename} -b bohrium -n {decomp} ' + STATIC_SETTINGS,
    'fortran': '{python} {filename} --fortran {fortran_library} ' + STATIC_SETTINGS,
    'fortran-mpi': '{mpiexec} -n {nproc} {python} {filename} --fortran {fortran_library} -n {decomp} ' + STATIC_SETTINGS
}
SLURM_COMMANDS = {
    'numpy': 'srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b numpy ' + STATIC_SETTINGS,
    'numpy-mpi': 'srun --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} -b numpy -n {decomp} ' + STATIC_SETTINGS,
    'bohrium': 'OMP_NUM_THREADS={nproc} BH_STACK=openmp BH_OPENMP_PROF=1 srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b bohrium ' + STATIC_SETTINGS,
    'bohrium-opencl': 'BH_STACK=opencl BH_OPENCL_PROF=1 srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b bohrium ' + STATIC_SETTINGS,
    'bohrium-cuda': 'BH_STACK=cuda BH_CUDA_PROF=1 srun --ntasks 1 --cpus-per-task {nproc} -- {python} {filename} -b bohrium ' + STATIC_SETTINGS,
    'bohrium-mpi': 'OMP_NUM_THREADS=1 srun --ntasks {nproc} --cpus-per-task 2 -- {python} {filename} -b bohrium -n {decomp} ' + STATIC_SETTINGS,
    'fortran': 'srun --ntasks 1 -- {python} {filename} --fortran {fortran_library} ' + STATIC_SETTINGS,
    'fortran-mpi': 'srun --ntasks {nproc} --cpus-per-task 1 -- {python} {filename} --fortran {fortran_library} -n {decomp} ' + STATIC_SETTINGS
}
AVAILABLE_BENCHMARKS = [f for f in os.listdir(TESTDIR) if f.endswith('_benchmark.py')]
TIME_PATTERN = r'Time step took ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)s'


def check_arguments(fortran_library, components, float_type, burnin, timesteps, **kwargs):
    fortran_version = check_fortran_library(fortran_library)

    if 'fortran' in components or 'fortran-mpi' in components:
        if not fortran_library:
            raise click.UsageError('Path to fortran library must be given when running fortran components')
        if not fortran_version:
            raise click.UsageError('Fortran library failed to import')

    if fortran_version != 'parallel' and 'fortran-mpi' in components:
        raise click.UsageError('Fortran library must be compiled with MPI support for fortran-mpi component')

    if float_type != 'float64' and ('fortran' in components or 'fortran-mpi' in components):
        raise click.UsageError('Can run Fortran components only with "float64" float type')

    if not burnin < timesteps:
        raise click.UsageError('burnin must be smaller than number of timesteps')


def check_fortran_library(path):
    if not path:
        return None

    def _check_library(module):
        spec = importlib.util.spec_from_file_location(module, path)
        try:
            importlib.util.module_from_spec(spec)
        except ImportError:
            return False
        else:
            return True

    if _check_library('pyOM_code'):
        return 'sequential'

    if _check_library('pyOM_code_MPI'):
        return 'parallel'

    return None


def _factorize(num):
    j = 2
    while num > 1:
        for i in range(j, int(math.sqrt(num + 0.05)) + 1):
            if num % i == 0:
                num /= i
                j = i
                yield i
                break
        else:
            if num > 1:
                yield num
                break


def _decompose_num(num, into=2):
    out = [1] * into
    for fac, i in zip(_factorize(num), itertools.cycle(range(into))):
        out[i] *= fac

    return tuple(map(int, out))


def _round_to_multiple(num, divisor):
    return int(round(num / divisor) * divisor)


@click.command('veros-benchmarks', help='Run Veros benchmarks')
@click.option('-f', '--fortran-library', type=str, help='Path to pyOM2 fortran library')
@click.option('-s', '--sizes', multiple=True, type=float, required=True,
              help='Problem sizes to test (total number of elements)')
@click.option('-c', '--components', multiple=True, type=click.Choice(COMPONENTS), default=['numpy'], metavar='COMPONENT',
              help='Numerical backend components to benchmark (possible values: {})'.format(', '.join(COMPONENTS)))
@click.option('-n', '--nproc', type=int, default=multiprocessing.cpu_count(),
              help='Number of processes / threads for parallel execution')
@click.option('-o', '--outfile', type=click.Path(exists=False), default='benchmark_{}.yaml'.format(time.time()),
              help='YAML file to write timings to')
@click.option('-t', '--timesteps', default=100, type=int, help='Number of time steps that each benchmark is run for')
@click.option('--only', multiple=True, default=AVAILABLE_BENCHMARKS,
              help='Run only these benchmarks (possible values: {})'.format(', '.join(AVAILABLE_BENCHMARKS)),
              type=click.Choice(AVAILABLE_BENCHMARKS), required=False, metavar='BENCHMARK')
@click.option('--mpiexec', default='mpiexec', help='Executable used for calling MPI (e.g. mpirun, mpiexec)')
@click.option('--slurm', is_flag=True, help='Run benchmarks using SLURM scheduling command (srun)')
@click.option('--debug', is_flag=True, help='Additionally print each command that is executed')
@click.option('--float-type', default='float64', help='Data type for floating point arrays in Veros components')
@click.option('--burnin', default=3, type=int, help='Number of iterations to exclude in timings')
def run(**kwargs):
    check_arguments(**kwargs)

    proc_decom = _decompose_num(kwargs['nproc'], 2)
    settings = {
        'timesteps': kwargs['timesteps'],
        'nproc': kwargs['nproc'],
        'float_type': kwargs['float_type'],
        'burnin': kwargs['burnin'],
        'decomp': '%s %s' % proc_decom,
        'fortran_library': kwargs['fortran_library'],
        'mpiexec': kwargs['mpiexec'],
    }

    out_data = {}
    all_passed = True
    try:
        for f in kwargs['only']:
            out_data[f] = []
            print('running benchmark {} '.format(f))
            for size in kwargs['sizes']:
                n = int(size ** (1. / 3.)) + 1
                nx = _round_to_multiple(2 * n, proc_decom[0])
                ny = _round_to_multiple(2 * n, proc_decom[1])
                nz = n // 4
                real_size = nx * ny * nz
                print(' current size: {}'.format(real_size))

                cmd_args = settings.copy()
                cmd_args.update({
                    'python': sys.executable,
                    'filename': os.path.realpath(os.path.join(TESTDIR, f)),
                    'nx': nx, 'ny': ny, 'nz': nz
                })

                for comp in kwargs['components']:
                    cmd = (SLURM_COMMANDS[comp] if kwargs['slurm'] else BENCHMARK_COMMANDS[comp]).format(**cmd_args)
                    if kwargs['debug']:
                        print('  $ ' + cmd)
                    sys.stdout.write('  {:<15} ... '.format(comp))
                    sys.stdout.flush()
                    try: # must run each benchmark in its own Python subprocess to reload the Fortran library
                        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        print('failed')
                        print(e.output.decode('utf-8'))
                        all_passed = False
                        continue
                    output = output.decode('utf-8')
                    iteration_times = list(map(float, re.findall(TIME_PATTERN, output)))[kwargs['burnin']:]
                    if not iteration_times:
                        raise RuntimeError('could not extract iteration times from output')
                    total_elapsed = sum(iteration_times)
                    print('{:>6.2f}s'.format(total_elapsed))

                    bohrium_stats = None
                    if 'bohrium' in comp:
                        bohrium_stats = {'Pre-fusion': None, 'Fusion': None,
                                         'Compilation': None, 'Exec': None,
                                         'Copy2dev': None, 'Copy2host': None,
                                         'Offload': None, 'Other': None}
                        patterns = {stat: r'\s*{}\:\s*(\d+\.?\d*)s'.format(stat) for stat in bohrium_stats.keys()}
                        for line in output.splitlines():
                            for stat in bohrium_stats.keys():
                                match = re.match(patterns[stat], line)
                                if match:
                                    bohrium_stats[stat] = float(match.group(1))

                    out_data[f].append({
                        'component': comp,
                        'size': real_size,
                        'wall_time': total_elapsed,
                        'per_iteration': {
                            'best': float(np.min(iteration_times)),
                            'worst': float(np.max(iteration_times)),
                            'mean': float(np.mean(iteration_times)),
                            'stdev': float(np.std(iteration_times)),
                        },
                        'bohrium_stats': bohrium_stats
                    })
    finally:
        with open(kwargs['outfile'], 'w') as f:
            yaml.dump({'benchmarks': out_data, 'settings': settings}, f, default_flow_style=False)

    raise SystemExit(int(not all_passed))


if __name__ == '__main__':
    run()
