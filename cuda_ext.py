import os

from Cython.Distutils import build_ext


# code taken from https://github.com/rmcgibbo/npcuda-example
# under BSD 2-Clause "Simplified" License


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'cuda_root', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME and CUDA_ROOT env variables.
    If not found, everything is based on finding 'nvcc' in the PATH.
    """

    # First check if any common env variable is in use
    if 'CUDAHOME' in os.environ:
        cuda_root = os.environ['CUDAHOME']
        nvcc = os.path.join(cuda_root, 'bin', 'nvcc')
    elif 'CUDA_ROOT' in os.environ:
        cuda_root = os.environ['CUDA_ROOT']
        nvcc = os.path.join(cuda_root, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is not None:
            cuda_root = os.path.dirname(os.path.dirname(nvcc))
        else:
            cuda_root = None

    if cuda_root is None:
        return {
            'cuda_root': None,
            'nvcc': None,
            'include': [],
            'lib64': [],
            'cflags': [],
        }

    cflags = [
        '--ptxas-options=-v', '-c',
        '--compiler-options', "'-fPIC'"
    ]

    cm = os.environ.get("CUDA_COMPUTE_CAPABILITY")
    if cm is not None:
        cflags.append(
            '-gencode=arch=compute_{cm},code=compute_{cm}'.format(cm=cm)
        )
    else:
        print(
            "Warning: Consider settings the CUDA_COMPUTE_CAPABILITY environment "
            "variable to your GPU's compute capability."
        )

    return {
        'cuda_root': cuda_root,
        'nvcc': nvcc,
        'include': [os.path.join(cuda_root, 'include')],
        'lib64': [os.path.join(cuda_root, 'lib64')],
        'cflags': cflags,
    }


def customize_compiler_for_nvcc(self):
    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    default_compile = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', cuda_info['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        default_compile(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


cuda_info = locate_cuda()
