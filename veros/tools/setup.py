from veros.core.operators import numpy as np
import numpy as onp

import scipy.interpolate
import scipy.spatial


def interpolate(coords, var, interp_coords, missing_value=None, fill=True, kind='linear'):
    """Interpolate globally defined data to a different (regular) grid.

    Arguments:
       coords: Tuple of coordinate arrays for each dimension.
       var (:obj:`ndarray` of dim (nx1, ..., nxd)): Variable data to interpolate.
       interp_coords: Tuple of coordinate arrays to interpolate to.
       missing_value (optional): Value denoting cells of missing data in ``var``.
          Is replaced by `NaN` before interpolating. Defaults to `None`, which means
          no replacement is taking place.
       fill (bool, optional): Whether `NaN` values should be replaced by the nearest
          finite value after interpolating. Defaults to ``True``.
       kind (str, optional): Order of interpolation. Supported are `nearest` and
          `linear` (default).

    Returns:
       :obj:`ndarray` containing the interpolated values on the grid spanned by
       ``interp_coords``.

    """
    if len(coords) != len(interp_coords) or len(coords) != var.ndim:
        raise ValueError('Dimensions of coordinates and values do not match')

    if missing_value is not None:
        invalid_mask = np.isclose(var, missing_value)
        var = np.where(invalid_mask, np.nan, var)

    if var.ndim > 1 and coords[0].ndim == 1:
        interp_grid = np.rollaxis(np.array(np.meshgrid(
            *interp_coords, indexing='ij')), 0, len(interp_coords) + 1)
    else:
        interp_grid = interp_coords

    coords = [onp.array(c) for c in coords]
    var = scipy.interpolate.interpn(coords, onp.array(var), interp_grid,
                                    bounds_error=False, fill_value=np.nan, method=kind)
    var = np.asarray(var)

    if fill:
        var = fill_holes(var)

    return var


def fill_holes(data):
    """A simple inpainting function that replaces NaN values in `data` with the
    nearest finite value.
    """
    data = onp.array(data)
    dim = data.ndim
    flag = ~onp.isnan(data)

    slcs = [slice(None)] * dim

    while np.any(~flag):
        for i in range(dim):
            slcs1 = slcs[:]
            slcs2 = slcs[:]
            slcs1[i] = slice(0, -1)
            slcs2[i] = slice(1, None)

            slcs1 = tuple(slcs1)
            slcs2 = tuple(slcs2)

            # replace from the right
            repmask = np.logical_and(~flag[slcs1], flag[slcs2])
            data[slcs1][repmask] = data[slcs2][repmask]
            flag[slcs1][repmask] = True

            # replace from the left
            repmask = np.logical_and(~flag[slcs2], flag[slcs1])
            data[slcs2][repmask] = data[slcs1][repmask]
            flag[slcs2][repmask] = True

    return np.asarray(data)


def get_periodic_interval(current_time, cycle_length, rec_spacing, n_rec):
    """Used for linear interpolation between periodic time intervals.

    One common application is the interpolation of external forcings that are defined
    at discrete times (e.g. one value per month of a standard year) to the current
    time step.

    Arguments:
       current_time (float): Time to interpolate to.
       cycle_length (float): Total length of one periodic cycle.
       rec_spacing (float): Time spacing between each data record.
       n_rec (int): Total number of records available.

    Returns:
       :obj:`tuple` containing (n1, f1), (n2, f2): Indices and weights for the interpolated
       record array.

    Example:
       The following interpolates a record array ``data`` containing 12 monthly values
       to the current time step:

       >>> year_in_seconds = 60. * 60. * 24. * 365.
       >>> current_time = 60. * 60. * 24. * 45. # mid-february
       >>> print(data.shape)
       (360, 180, 12)
       >>> (n1, f1), (n2, f2) = get_periodic_interval(current_time, year_in_seconds, year_in_seconds / 12, 12)
       >>> data_at_current_time = f1 * data[..., n1] + f2 * data[..., n2]

    """
    current_time = current_time % cycle_length
    # using np.array works with both NumPy and JAX
    t_idx_1 = np.array(current_time // rec_spacing, dtype='int')
    t_idx_2 = np.array((1 + t_idx_1) % n_rec, dtype='int')
    weight_2 = (current_time - rec_spacing * t_idx_1) / rec_spacing
    weight_1 = 1.0 - weight_2
    return (t_idx_1, weight_1), (t_idx_2, weight_2)


def make_cyclic(longitude, array=None, wrap=360.):
    """Create a cyclic version of a longitude array and (optionally) another array.

    Arguments:
        longitude (ndarray): Longitude array of shape (nlon, ...).
        array (ndarray): Another array that is to be made cyclic of shape (nlon, ...).
        wrap (float): Wrapping value, defaults to 360 (degrees).

    Returns:
        Tuple containing (cyclic_longitudes, cyclic_array) if `array` is given, otherwise
        just the ndarray cyclic_longitudes of shape (2 * nlon, ...).

    """
    lonsize = longitude.shape[0]
    cyclic_longitudes = np.hstack((longitude[lonsize//2:, ...] - wrap, longitude, longitude[:lonsize//2, ...] + wrap))
    if array is None:
        return cyclic_longitudes
    cyclic_array = np.hstack((array[lonsize//2:, ...], array, array[:lonsize//2, ...]))
    return cyclic_longitudes, cyclic_array


def get_coastline_distance(coords, coast_mask, spherical=False, radius=None, num_candidates=None, n_jobs=-1):
    """Calculate the (approximate) distance of each water cell from the nearest coastline.

    Arguments:
        coords (tuple of ndarrays): Tuple containing x and y (longitude and latitude)
            coordinate arrays of shape (nx, ny).
        coast_mask (ndarray): Boolean mask indicating whether a cell is a land cell
            (must be same shape as coordinate arrays).
        spherical (bool): Use spherical instead of Cartesian coordinates.
            When this is `True`, cyclical boundary conditions are used, and the
            resulting distances are only approximate. Cells are pre-sorted by
            Euclidean lon-lat distance, and great circle distances are calculated for
            the first `num_candidates` elements. Defaults to `False`.
        radius (float): Radius of spherical coordinate system. Must be given when
            `spherical` is `True`.
        num_candidates (int): Number of candidates to calculate great circle distances
            for for each water cell. The higher this value, the more accurate the returned
            distances become when `spherical` is `True`. Defaults to the square root
            of the number of coastal cells.
        n_jobs (int): Number of parallel jobs to determine nearest neighbors
            (defaults to -1, which uses all available threads).

    Returns:
        :obj:`ndarray` of shape (nx, ny) indicating the distance to the nearest land
        cell (0 if cell is land).

    Example:
        The following returns coastal distances of all T cells for a spherical Veros setup.

        >>> coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], indexing='ij')
        >>> dist = tools.get_coastline_distance(coords, self.kbot > 0, spherical=True, radius=self.radius)

    """
    if not len(coords) == 2:
        raise ValueError('coords must be lon-lat tuple')
    if not all(c.shape == coast_mask.shape for c in coords):
        raise ValueError('coordinates must have same shape as coastal mask')
    if spherical and not radius:
        raise ValueError('radius must be given for spherical coordinates')

    watercoords = onp.array([c[~coast_mask] for c in coords]).T
    if spherical:
        coastcoords = onp.array(make_cyclic(coords[0][coast_mask], coords[1][coast_mask])).T
    else:
        coastcoords = onp.array((coords[0][coast_mask], coords[1][coast_mask])).T
    coast_kdtree = scipy.spatial.cKDTree(coastcoords)

    distance = onp.zeros(coords[0].shape)

    if spherical:
        def spherical_distance(coords1, coords2):
            """Calculate great circle distance from latitude and longitude"""
            coords1 *= onp.pi / 180.
            coords2 *= onp.pi / 180.
            lon1, lon2, lat1, lat2 = coords1[..., 0], coords2[..., 0], coords1[..., 1], coords2[..., 1]
            return radius * onp.arccos(onp.sin(lat1) * onp.sin(lat2) + onp.cos(lat1) * onp.cos(lat2) * onp.cos(lon1 - lon2))

        if not num_candidates:
            num_candidates = int(onp.sqrt(onp.count_nonzero(~coast_mask)))
        i_nearest = coast_kdtree.query(watercoords, k=num_candidates, n_jobs=n_jobs)[1]
        approx_nearest = coastcoords[i_nearest]
        distance[~coast_mask] = onp.min(spherical_distance(approx_nearest, watercoords[..., onp.newaxis, :]), axis=-1)

    else:
        distance[~coast_mask] = coast_kdtree.query(watercoords, n_jobs=n_jobs)[0]

    return np.asarray(distance)


def get_uniform_grid_steps(total_length, stepsize):
    """Get uniform grid step sizes in an interval.

    Arguments:
        total_length (float): total length of the resulting grid
        stepsize (float): grid step size

    Returns:
        :obj:`ndarray` of grid steps

    Example:
        >>> uniform_steps = uniform_grid_setup(6., 0.25)
        >>> uniform_steps
        [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25 ]

    """
    if total_length % stepsize:
        raise ValueError('total length must be an integer multiple of stepsize')
    return stepsize * np.ones(int(total_length / stepsize))


def get_stretched_grid_steps(n_cells, total_length, minimum_stepsize, stretching_factor=2.5,
                             two_sided_grid=False, refine_towards='upper'):
    """Computes stretched grid steps for regional and global domains with either
    one or two-sided stretching using a hyperbolic tangent stretching function.

    Arguments:
        n_cells (int): Number of grid points.
        total_length (float): Length of the grid interval to be covered (sum of the
            resulting grid steps).
        minimum_stepsize (float): Grid step size on the lower end of the interval.
        stretching_factor (float, optional): Coefficient of the `tanh` stretching
            function. The higher this value, the more abrupt the step sizes change.
        two_sided_grid (bool, optional): If set to `True`, the resulting grid will be symmetrical
            around the center. Defaults to `False`.
        refine_towards ('upper' or 'lower', optional): The side of the interval that is to be refined.
            Defaults to 'upper'.

    Returns:
        :obj:`ndarray` of shape `(n_cells)` containing grid steps.

    Examples:
        >>> dyt = get_stretched_grid_steps(14, 180, 5)
        >>> dyt
        [  5.10517337   5.22522948   5.47813251   5.99673813   7.00386752
           8.76808565  11.36450896  14.34977676  16.94620006  18.71041819
          19.71754758  20.2361532   20.48905624  20.60911234]
        >>> dyt.sum()
        180.0

        >>> dyt = get_stretched_grid_steps(14, 180, 5, stretching_factor=4.)
        >>> dyt
        [  5.00526979   5.01802837   5.06155549   5.20877528   5.69251688
           7.14225176  10.51307232  15.20121339  18.57203395  20.02176884
          20.50551044  20.65273022  20.69625734  20.70901593]
        >>> dyt.sum()
        180.0

    """

    if refine_towards not in ('upper', 'lower'):
        raise ValueError('refine_towards must be "upper" or "lower"')
    if two_sided_grid:
        if n_cells % 2:
            raise ValueError('number of grid points must be even integer number (given: {})'.format(n_cells))
        n_cells = n_cells / 2

    stretching_function = np.tanh(stretching_factor * np.linspace(-1, 1, n_cells))

    if refine_towards == 'lower':
        stretching_function = stretching_function[::-1]
    if two_sided_grid:
        stretching_function = np.concatenate((stretching_function[::-1], stretching_function))

    def normalize_sum(var, sum_value, minimum_value=0.):
        if abs(var.sum()) < 1e-5:
            var += 1
        var *= (sum_value - len(var) * minimum_value) / var.sum()
        return var + minimum_value

    stretching_function = normalize_sum(stretching_function, total_length, minimum_stepsize)
    assert abs(1 - np.sum(stretching_function) / total_length) < 1e-5, 'precision error'
    return stretching_function


def get_vinokur_grid_steps(n_cells, total_length, lower_stepsize, upper_stepsize=None,
                           two_sided_grid=False, refine_towards='upper'):
    """Computes stretched grid steps for regional and global domains with either
    one or two-sided stretching using Vinokur stretching.

    This stretching function minimizes discretization errors on finite difference
    grids.

    Arguments:
        n_cells (int): Number of grid points.
        total_length (float): Length of the grid interval to be covered (sum of the
            resulting grid steps).
        lower_stepsize (float): Grid step size on the lower end of the interval.
        upper_stepsize (float or ``None``, optional): Grid step size on the upper end of the interval.
            If not given, the one-sided version of the algorithm is used (that enforces zero curvature
            on the upper end).
        two_sided_grid (bool, optional): If set to `True`, the resulting grid will be symmetrical
            around the center. Defaults to `False`.
        refine_towards ('upper' or 'lower', optional): The side of the interval that is to be refined.
            Defaults to 'upper'.

    Returns:
        :obj:`ndarray` of shape `(n_cells)` containing grid steps.

    Reference:
        Vinokur, Marcel, On One-Dimensional Stretching Functions for Finite-Difference Calculations,
        Journal of Computational Physics. 50, 215, 1983.

    Examples:
        >>> dyt = get_vinokur_grid_steps(14, 180, 5, two_sided_grid=True)
        >>> dyt
        [ 18.2451554   17.23915939  15.43744632  13.17358802  10.78720589
           8.53852027   6.57892471   6.57892471   8.53852027  10.78720589
          13.17358802  15.43744632  17.23915939  18.2451554 ]
        >>> dyt.sum()
        180.

        >>> dyt = get_vinokur_grid_steps(14, 180, 5, upper_stepsize=10)
        >>> dyt
        [  5.9818365    7.3645667    8.92544833  10.61326984  12.33841985
          13.97292695  15.36197306  16.3485688   16.80714121  16.67536919
          15.97141714  14.78881918  13.27136448  11.57887877 ]
        >>> dyt.sum()
        180.

    """
    if refine_towards not in ('upper', 'lower'):
        raise ValueError('refine_towards must be "upper" or "lower"')
    if two_sided_grid:
        if n_cells % 2:
            raise ValueError('number of grid points must be an even integer (given: {})'.format(n_cells))
        n_cells = n_cells // 2

    n_cells += 1

    def approximate_sinc_inverse(y):
        """Approximate inverse of sin(y) / y"""
        if y < 0.26938972:
            inv = np.pi * (1 - y + y**2 - (1 + np.pi**2 / 6) * y**3 + 6.794732 * y**4 - 13.205501 * y**5 + 11.726095 * y**6)
        else:
            ybar = 1. - y
            inv = np.sqrt(6 * ybar) * (1 + .15 * ybar + 0.057321429 * ybar**2 + 0.048774238 * ybar**3 - 0.053337753 * ybar**4 + 0.075845134 * ybar**5)
        assert abs(1 - np.sin(inv) / inv / y) < 1e-2, 'precision error'
        return inv

    def approximate_sinhc_inverse(y):
        """Approximate inverse of sinh(y) / y"""
        if y < 2.7829681:
            ybar = y - 1.
            inv = np.sqrt(6 * ybar) * (1 - 0.15 * ybar + 0.057321429 * ybar**2 - 0.024907295 * ybar**3 + 0.0077424461 * ybar**4 - 0.0010794123 * ybar**5)
        else:
            v = np.log(y)
            w = 1. / y - 0.028527431
            inv = v + (1 + 1. / v) * np.log(2 * v) - 0.02041793 + 0.24902722 * w + 1.9496443 * w**2 - 2.6294547 * w**3 + 8.56795911 * w**4
        assert abs(1 - np.sinh(inv) / inv / y) < 1e-2, 'precision error'
        return inv

    target_sum = total_length
    if two_sided_grid:
        target_sum *= .5

    s0 = float(target_sum) / float(lower_stepsize * n_cells)
    if upper_stepsize:
        s1 = float(target_sum) / float(upper_stepsize * n_cells)
        a, b = np.sqrt(s1 / s0), np.sqrt(s1 * s0)
        if b > 1:
            stretching_factor = approximate_sinhc_inverse(b)
            stretched_grid = .5 + .5 * np.tanh(stretching_factor * np.linspace(-.5, .5, n_cells)) / np.tanh(.5 * stretching_factor)
        else:
            stretching_factor = approximate_sinc_inverse(b)
            stretched_grid = .5 + .5 * np.tan(stretching_factor * np.linspace(-.5, .5, n_cells)) / np.tan(.5 * stretching_factor)
        stretched_grid = stretched_grid / (a + (1. - a) * stretched_grid)
    else:
        if s0 > 1:
            stretching_factor = approximate_sinhc_inverse(s0) * .5
            stretched_grid = 1 + np.tanh(stretching_factor * np.linspace(0., 1., n_cells)) / np.tanh(stretching_factor)
        else:
            stretching_factor = approximate_sinc_inverse(s0) * .5
            stretched_grid = 1 + np.tan(stretching_factor * np.linspace(0., 1., n_cells)) / np.tan(stretching_factor)

    stretched_grid_steps = np.diff(stretched_grid * target_sum)
    if refine_towards == 'upper':
        stretched_grid_steps = stretched_grid_steps[::-1]
    if two_sided_grid:
        stretched_grid_steps = np.concatenate((stretched_grid_steps[::-1], stretched_grid_steps))

    assert abs(1 - np.sum(stretched_grid_steps) / total_length) < 1e-5, 'precision error'
    return stretched_grid_steps
