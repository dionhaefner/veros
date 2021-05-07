import numpy as np

from test_base import VerosPyOMUnitTest
from veros.core import numerics


class TridiagTest(VerosPyOMUnitTest):
    nx, ny, nz = 100, 100, 100

    def initialize(self):
        pass

    def run(self):
        a, b, c, d = (np.random.randn(self.nx, self.ny, self.nz) for _ in range(4))

        out_legacy = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nx):
            for j in range(self.ny):
                out_legacy[i, j] = self.veros_legacy.call_fortran_routine(
                    'solve_tridiag', a=a[i, j], b=b[i, j], c=c[i, j], d=d[i, j], n=self.nz
                )

        out_new = numerics.solve_tridiag(self.veros_new.state, a, b, c, d)
        np.testing.assert_allclose(out_legacy, out_new)


def test_tridiag(pyom2_lib, backend):
    TridiagTest(fortran=pyom2_lib, backend=backend).run()
