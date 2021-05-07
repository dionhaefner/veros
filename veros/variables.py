import numpy as np

from veros import runtime_settings


class Variable:
    def __init__(self, name, dims, units="", long_description="", dtype=None,
                 output=False, time_dependent=True, scale=1.,
                 write_to_restart=False, extra_attributes=None, mask=None,
                 active=True, initial=None):
        if dims is not None:
            dims = tuple(dims)

        self.name = name
        self.dims = dims
        self.units = units
        self.long_description = long_description
        self.dtype = dtype
        self.output = output
        self.time_dependent = time_dependent
        self.scale = scale
        self.write_to_restart = write_to_restart
        self.active = active
        self.initial = initial

        self.get_mask = lambda vs: None

        if mask is not None:
            if not callable(mask):
                raise TypeError('mask argument has to be callable')
            self.get_mask = mask
        elif dims is not None:
            if dims[:3] in DEFAULT_MASKS:
                self.get_mask = DEFAULT_MASKS[dims[:3]]
            elif dims[:2] in DEFAULT_MASKS:
                self.get_mask = DEFAULT_MASKS[dims[:2]]

        #: Additional attributes to be written in netCDF output
        self.extra_attributes = extra_attributes or {}

    def __repr__(self):
        attr_str = []
        for v in vars(self):
            attr_str.append(f'{v}={getattr(self, v)}')
        attr_str = ', '.join(attr_str)
        return f'{self.__class__.__qualname__}({attr_str})'


# fill value for netCDF output (invalid data is replaced by this value)
FILL_VALUE = -1e18

#
XT = ('xt',)
XU = ('xu',)
YT = ('yt',)
YU = ('yu',)
ZT = ('zt',)
ZW = ('zw',)
T_HOR = ('xt', 'yt')
U_HOR = ('xu', 'yt')
V_HOR = ('xt', 'yu')
ZETA_HOR = ('xu', 'yu')
T_GRID = ('xt', 'yt', 'zt')
U_GRID = ('xu', 'yt', 'zt')
V_GRID = ('xt', 'yu', 'zt')
W_GRID = ('xt', 'yt', 'zw')
ZETA_GRID = ('xu', 'yu', 'zt')
TIMESTEPS = ('timesteps',)
ISLE = ('isle',)
TENSOR_COMP = ('tensor1', 'tensor2')

# those are written to netCDF output by default
BASE_DIMENSIONS = XT + XU + YT + YU + ZT + ZW + ISLE
GHOST_DIMENSIONS = ('xt', 'yt', 'xu', 'yu')

# these are the settings that are getting used to determine shapes
DIM_TO_SHAPE_VAR = {
    'xt': 'nx',
    'xu': 'nx',
    'yt': 'ny',
    'yu': 'ny',
    'zt': 'nz',
    'zw': 'nz',
    'timesteps': 3,
    'tensor1': 2,
    'tensor2': 2,
    'isle': 0,
}

DEFAULT_MASKS = {
    T_HOR: lambda vs: vs.maskT[:, :, -1],
    U_HOR: lambda vs: vs.maskU[:, :, -1],
    V_HOR: lambda vs: vs.maskV[:, :, -1],
    ZETA_HOR: lambda vs: vs.maskZ[:, :, -1],
    T_GRID: lambda vs: vs.maskT,
    U_GRID: lambda vs: vs.maskU,
    V_GRID: lambda vs: vs.maskV,
    W_GRID: lambda vs: vs.maskW,
    ZETA_GRID: lambda vs: vs.maskZ
}

# custom mask for streamfunction
ZETA_HOR_ERODED = lambda vs: vs.maskZ[:, :, -1] | vs.boundary_mask.sum(axis=2)  # noqa: E731


def get_shape(dimensions, grid, include_ghosts=True, local=True):
    from veros.routines import CURRENT_CONTEXT
    from veros.distributed import SCATTERED_DIMENSIONS

    if grid is None:
        return ()

    px, py = runtime_settings.num_proc
    grid_shapes = dimensions.copy()

    if local and CURRENT_CONTEXT.is_dist_safe:
        for pxi, dims in zip((px, py), SCATTERED_DIMENSIONS):
            for dim in dims:
                if dim not in grid_shapes:
                    continue

                grid_shapes[dim] = grid_shapes[dim] // pxi

    if include_ghosts:
        for d in GHOST_DIMENSIONS:
            if d in grid_shapes:
                grid_shapes[d] += 4

    shape = []
    for grid_dim in grid:
        if grid_dim in grid_shapes:
            shape.append(grid_shapes[grid_dim])
        elif isinstance(grid_dim, int):
            shape.append(grid_dim)
        elif hasattr(dimensions, grid_dim):
            shape.append(getattr(dimensions, grid_dim))
        else:
            raise ValueError('unrecognized dimension %s' % grid_dim)

    return tuple(shape)


def remove_ghosts(array, dims):
    if dims is None:
        # scalar
        return array

    ghost_mask = tuple(slice(2, -2) if dim in GHOST_DIMENSIONS else slice(None) for dim in dims)
    return array[ghost_mask]


def add_ghosts(vs, array, dims):
    full_shape = tuple([i + 4 if dim in GHOST_DIMENSIONS else i for i,
                        dim in zip(array.shape, dims)])
    newarr = np.zeros(full_shape, dtype=array.dtype)
    ghost_mask = tuple(slice(2, -2) if dim in GHOST_DIMENSIONS else slice(None) for dim in dims)
    newarr[ghost_mask] = array
    return newarr


VARIABLES = {
    # scalars
    'tau': Variable(
        'Index of current time step', None, '',
        'Index of current time step', dtype='int', initial=1
    ),
    'taup1': Variable(
        'Index of next time step', None, '',
        'Index of next time step', dtype='int', initial=2
    ),
    'taum1': Variable(
        'Index of last time step', None, '',
        'Index of last time step', dtype='int', initial=0
    ),
    'time': Variable(
        'Current time', None, '',
        'Current time',
    ),
    'itt': Variable(
        'Current iteration', None, '',
        'Current iteration', dtype='int', initial=0
    ),

    # base variables
    'dxt': Variable(
        'Zonal T-grid spacing', XT, 'm',
        'Zonal (x) spacing of T-grid point',
        output=True, time_dependent=False
    ),
    'dxu': Variable(
        'Zonal U-grid spacing', XU, 'm',
        'Zonal (x) spacing of U-grid point',
        output=True, time_dependent=False
    ),
    'dyt': Variable(
        'Meridional T-grid spacing', YT, 'm',
        'Meridional (y) spacing of T-grid point',
        output=True, time_dependent=False
    ),
    'dyu': Variable(
        'Meridional U-grid spacing', YU, 'm',
        'Meridional (y) spacing of U-grid point',
        output=True, time_dependent=False
    ),
    'zt': Variable(
        'Vertical coordinate (T)', ZT, 'm', 'Vertical coordinate',
        output=True, time_dependent=False, extra_attributes={'positive': 'up'}
    ),
    'zw': Variable(
        'Vertical coordinate (W)', ZW, 'm', 'Vertical coordinate', output=True,
        time_dependent=False, extra_attributes={'positive': 'up'}
    ),
    'dzt': Variable(
        'Vertical spacing (T)', ZT, 'm', 'Vertical spacing', output=True, time_dependent=False
    ),
    'dzw': Variable(
        'Vertical spacing (W)', ZW, 'm', 'Vertical spacing', output=True, time_dependent=False
    ),
    'cost': Variable(
        'Metric factor (T)', YT, '1', 'Metric factor for spherical coordinates',
        time_dependent=False
    ),
    'cosu': Variable(
        'Metric factor (U)', YU, '1', 'Metric factor for spherical coordinates',
        time_dependent=False
    ),
    'tantr': Variable(
        'Metric factor', YT, '1', 'Metric factor for spherical coordinates',
        time_dependent=False
    ),
    'coriolis_t': Variable(
        'Coriolis frequency', T_HOR, '1/s',
        'Coriolis frequency at T grid point', time_dependent=False
    ),

    'kbot': Variable(
        'Index of deepest cell', T_HOR, '',
        'Index of the deepest grid cell (counting from 1, 0 means all land)',
        dtype='int', time_dependent=False
    ),
    'ht': Variable(
        'Total depth (T)', T_HOR, 'm', 'Total depth of the water column', output=True,
        time_dependent=False
    ),
    'hu': Variable(
        'Total depth (U)', U_HOR, 'm', 'Total depth of the water column', output=True,
        time_dependent=False
    ),
    'hv': Variable(
        'Total depth (V)', V_HOR, 'm', 'Total depth of the water column', output=True,
        time_dependent=False
    ),
    'hur': Variable(
        'Total depth (U), masked', U_HOR, 'm',
        'Total depth of the water column (masked)', time_dependent=False
    ),
    'hvr': Variable(
        'Total depth (V), masked', V_HOR, 'm',
        'Total depth of the water column (masked)', time_dependent=False
    ),
    'beta': Variable(
        'Change of Coriolis freq.', T_HOR, '1/(ms)',
        'Change of Coriolis frequency with latitude', output=True, time_dependent=False
    ),
    'area_t': Variable(
        'Area of T-box', T_HOR, 'm^2', 'Area of T-box', output=True, time_dependent=False
    ),
    'area_u': Variable(
        'Area of U-box', U_HOR, 'm^2', 'Area of U-box', output=True, time_dependent=False
    ),
    'area_v': Variable(
        'Area of V-box', V_HOR, 'm^2', 'Area of V-box', output=True, time_dependent=False
    ),

    'maskT': Variable(
        'Mask for tracer points', T_GRID, '',
        'Mask in physical space for tracer points', time_dependent=False, dtype='bool',
    ),
    'maskU': Variable(
        'Mask for U points', U_GRID, '',
        'Mask in physical space for U points', time_dependent=False, dtype='bool',
    ),
    'maskV': Variable(
        'Mask for V points', V_GRID, '',
        'Mask in physical space for V points', time_dependent=False, dtype='bool',
    ),
    'maskW': Variable(
        'Mask for W points', W_GRID, '',
        'Mask in physical space for W points', time_dependent=False, dtype='bool',
    ),
    'maskZ': Variable(
        'Mask for Zeta points', ZETA_GRID, '',
        'Mask in physical space for Zeta points', time_dependent=False, dtype='bool',
    ),

    'rho': Variable(
        'Density', T_GRID + TIMESTEPS, 'kg/m^3',
        'In-situ density anomaly, relative to the surface mean value of 1024 kg/m^3',
        output=True, write_to_restart=True
    ),

    'prho': Variable(
        'Potential density', T_GRID, 'kg/m^3',
        'Potential density anomaly, relative to the surface mean value of 1024 kg/m^3 '
        '(identical to in-situ density anomaly for equation of state type 1, 2, and 4)',
        output=True
    ),

    'int_drhodT': Variable(
        'Der. of dyn. enthalpy by temperature', T_GRID + TIMESTEPS, '?',
        'Partial derivative of dynamic enthalpy by temperature', output=True,
        write_to_restart=True
    ),
    'int_drhodS': Variable(
        'Der. of dyn. enthalpy by salinity', T_GRID + TIMESTEPS, '?',
        'Partial derivative of dynamic enthalpy by salinity', output=True,
        write_to_restart=True
    ),
    'Nsqr': Variable(
        'Square of stability frequency', W_GRID + TIMESTEPS, '1/s^2',
        'Square of stability frequency', output=True, write_to_restart=True
    ),
    'Hd': Variable(
        'Dynamic enthalpy', T_GRID + TIMESTEPS, 'm^2/s^2', 'Dynamic enthalpy',
        output=True, write_to_restart=True
    ),
    'dHd': Variable(
        'Change of dyn. enth. by adv.', T_GRID + TIMESTEPS, 'm^2/s^3',
        'Change of dynamic enthalpy due to advection', write_to_restart=True
    ),

    'temp': Variable(
        'Temperature', T_GRID + TIMESTEPS, 'deg C',
        'Conservative temperature', output=True, write_to_restart=True
    ),
    'dtemp': Variable(
        'Temperature tendency', T_GRID + TIMESTEPS, 'deg C/s',
        'Conservative temperature tendency', write_to_restart=True
    ),
    'salt': Variable(
        'Salinity', T_GRID + TIMESTEPS, 'g/kg', 'Salinity', output=True,
        write_to_restart=True
    ),
    'dsalt': Variable(
        'Salinity tendency', T_GRID + TIMESTEPS, 'g/(kg s)',
        'Salinity tendency', write_to_restart=True
    ),
    'dtemp_vmix': Variable(
        'Change of temp. by vertical mixing', T_GRID, 'deg C/s',
        'Change of temperature due to vertical mixing',
    ),
    'dtemp_hmix': Variable(
        'Change of temp. by horizontal mixing', T_GRID, 'deg C/s',
        'Change of temperature due to horizontal mixing',
    ),
    'dsalt_vmix': Variable(
        'Change of sal. by vertical mixing', T_GRID, 'deg C/s',
        'Change of salinity due to vertical mixing',
    ),
    'dsalt_hmix': Variable(
        'Change of sal. by horizontal mixing', T_GRID, 'deg C/s',
        'Change of salinity due to horizontal mixing',
    ),
    'dtemp_iso': Variable(
        'Change of temp. by isop. mixing', T_GRID, 'deg C/s',
        'Change of temperature due to isopycnal mixing plus skew mixing',
    ),
    'dsalt_iso': Variable(
        'Change of sal. by isop. mixing', T_GRID, 'deg C/s',
        'Change of salinity due to isopycnal mixing plus skew mixing',

    ),
    'forc_temp_surface': Variable(
        'Surface temperature flux', T_HOR, 'm K/s', 'Surface temperature flux',
        output=True
    ),
    'forc_salt_surface': Variable(
        'Surface salinity flux', T_HOR, 'm g/s kg', 'Surface salinity flux',
        output=True
    ),

    'u': Variable(
        'Zonal velocity', U_GRID + TIMESTEPS, 'm/s', 'Zonal velocity',
        output=True, write_to_restart=True
    ),
    'v': Variable(
        'Meridional velocity', V_GRID + TIMESTEPS, 'm/s', 'Meridional velocity',
        output=True, write_to_restart=True
    ),
    'w': Variable(
        'Vertical velocity', W_GRID + TIMESTEPS, 'm/s', 'Vertical velocity',
        output=True, write_to_restart=True
    ),
    'du': Variable(
        'Zonal velocity tendency', U_GRID + TIMESTEPS, 'm/s',
        'Zonal velocity tendency', write_to_restart=True
    ),
    'dv': Variable(
        'Meridional velocity tendency', V_GRID + TIMESTEPS, 'm/s',
        'Meridional velocity tendency', write_to_restart=True
    ),
    'du_cor': Variable(
        'Change of u by Coriolis force', U_GRID, 'm/s^2',
        'Change of u due to Coriolis force'
    ),
    'dv_cor': Variable(
        'Change of v by Coriolis force', V_GRID, 'm/s^2',
        'Change of v due to Coriolis force'
    ),
    'du_mix': Variable(
        'Change of u by vertical mixing', U_GRID, 'm/s^2',
        'Change of u due to implicit vertical mixing'
    ),
    'dv_mix': Variable(
        'Change of v by vertical mixing', V_GRID, 'm/s^2',
        'Change of v due to implicit vertical mixing'
    ),
    'du_adv': Variable(
        'Change of u by advection', U_GRID, 'm/s^2',
        'Change of u due to advection'
    ),
    'dv_adv': Variable(
        'Change of v by advection', V_GRID, 'm/s^2',
        'Change of v due to advection'
    ),
    'p_hydro': Variable(
        'Hydrostatic pressure', T_GRID, 'm^2/s^2', 'Hydrostatic pressure', output=True
    ),
    'kappaM': Variable(
        'Vertical viscosity', T_GRID, 'm^2/s', 'Vertical viscosity', output=True
    ),
    'kappaH': Variable(
        'Vertical diffusivity', W_GRID, 'm^2/s', 'Vertical diffusivity', output=True
    ),
    'surface_taux': Variable(
        'Surface wind stress', U_HOR, 'N/m^2', 'Zonal surface wind stress', output=True,
    ),
    'surface_tauy': Variable(
        'Surface wind stress', V_HOR, 'N/m^2', 'Meridional surface wind stress', output=True,
    ),
    'forc_rho_surface': Variable(
        'Surface density flux', T_HOR, '?', 'Surface potential density flux', output=True
    ),

    'psi': Variable(
        'Streamfunction', ZETA_HOR + TIMESTEPS, 'm^3/s', 'Barotropic streamfunction',
        output=True, write_to_restart=True, mask=ZETA_HOR_ERODED
    ),
    'dpsi': Variable(
        'Streamfunction tendency', ZETA_HOR + TIMESTEPS, 'm^3/s^2',
        'Streamfunction tendency', write_to_restart=True
    ),
    'land_map': Variable(
        'Land map', T_HOR, '', 'Land map', dtype='int32'
    ),
    'isle': Variable(
        'Island number', ISLE, '', 'Island number', output=True
    ),
    'psin': Variable(
        'Boundary streamfunction', ZETA_HOR + ISLE, 'm^3/s',
        'Boundary streamfunction', output=True, time_dependent=False,
        mask=ZETA_HOR_ERODED
    ),
    'dpsin': Variable(
        'Boundary streamfunction factor', ISLE + TIMESTEPS, '?',
        'Boundary streamfunction factor', write_to_restart=True
    ),
    'line_psin': Variable(
        'Boundary line integrals', ISLE + ISLE, '?',
        'Boundary line integrals', time_dependent=False
    ),
    'boundary_mask': Variable(
        'Boundary mask', T_HOR + ISLE, '',
        'Boundary mask', time_dependent=False, dtype='bool'
    ),
    'line_dir_south_mask': Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False, dtype='bool'
    ),
    'line_dir_north_mask': Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False, dtype='bool'
    ),
    'line_dir_east_mask': Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False, dtype='bool'
    ),
    'line_dir_west_mask': Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False, dtype='bool'
    ),

    'K_gm': Variable(
        'Skewness diffusivity', W_GRID, 'm^2/s',
        'GM diffusivity, either constant or from EKE model'
    ),
    'K_iso': Variable(
        'Isopycnal diffusivity', W_GRID, 'm^2/s', 'Along-isopycnal diffusivity'
    ),

    'K_diss_v': Variable(
        'Dissipation of kinetic Energy', W_GRID, 'm^2/s^3',
        'Kinetic energy dissipation by vertical, rayleigh and bottom friction',
        write_to_restart=True
    ),
    'K_diss_bot': Variable(
        'Dissipation of kinetic Energy', W_GRID, 'm^2/s^3',
        'Mean energy dissipation by bottom and rayleigh friction'
    ),
    'K_diss_h': Variable(
        'Dissipation of kinetic Energy', W_GRID, 'm^2/s^3',
        'Kinetic energy dissipation by horizontal friction'
    ),
    'K_diss_gm': Variable(
        'Dissipation of mean energy', W_GRID, 'm^2/s^3',
        'Mean energy dissipation by GM (TRM formalism only)',
    ),
    'P_diss_v': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by vertical diffusion'
    ),
    'P_diss_nonlin': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by nonlinear equation of state'
    ),
    'P_diss_iso': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by isopycnal mixing'
    ),
    'P_diss_skew': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by GM (w/o TRM)'
    ),
    'P_diss_hmix': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by horizontal mixing'
    ),
    'P_diss_adv': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by advection'
    ),
    'P_diss_sources': Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by external sources (e.g. restoring zones)',
    ),

    'u_wgrid': Variable(
        'U on W grid', W_GRID, 'm/s', 'Zonal velocity interpolated to W grid points'
    ),
    'v_wgrid': Variable(
        'V on W grid', W_GRID, 'm/s', 'Meridional velocity interpolated to W grid points'
    ),
    'w_wgrid': Variable(
        'W on W grid', W_GRID, 'm/s', 'Vertical velocity interpolated to W grid points'
    ),
    'xt': Variable(
        'Zonal coordinate (T)', XT,
        lambda settings: 'degrees_east' if settings.coord_degree else 'km',
        'Zonal (x) coordinate of T-grid point',
        output=True, time_dependent=False,
        scale=lambda settings: 1 if settings.coord_degree else 1e-3,
    ),
    'xu': Variable(
        'Zonal coordinate (U)', XU,
        lambda settings: 'degrees_east' if settings.coord_degree else 'km',
        'Zonal (x) coordinate of U-grid point',
        output=True, time_dependent=False,
        scale=lambda settings: 1 if settings.coord_degree else 1e-3,
    ),
    'yt': Variable(
        'Meridional coordinate (T)', YT,
        lambda settings: 'degrees_north' if settings.coord_degree else 'km',
        'Meridional (y) coordinate of T-grid point',
        output=True, time_dependent=False,
        scale=lambda settings: 1 if settings.coord_degree else 1e-3,
    ),
    'yu': Variable(
        'Meridional coordinate (U)', YU,
        lambda settings: 'degrees_north' if settings.coord_degree else 'km',
        'Meridional (y) coordinate of U-grid point',
        output=True, time_dependent=False,
        scale=lambda settings: 1 if settings.coord_degree else 1e-3,
    ),
    'temp_source': Variable(
        'Source of temperature', T_GRID, 'K/s',
        'Non-conservative source of temperature', output=True,
        active=lambda settings: settings.enable_tempsalt_sources,
    ),
    'salt_source': Variable(
        'Source of salt', T_GRID, 'g/(kg s)',
        'Non-conservative source of salt', output=True,
        active=lambda settings: settings.enable_tempsalt_sources,
    ),

    'u_source': Variable(
        'Source of zonal velocity', U_GRID, 'm/s^2',
        'Non-conservative source of zonal velocity', output=True,
        active=lambda settings: settings.enable_momentum_sources,
    ),
    'v_source': Variable(
        'Source of meridional velocity', V_GRID, 'm/s^2',
        'Non-conservative source of meridional velocity', output=True,
        active=lambda settings: settings.enable_momentum_sources,
    ),

    'K_11': Variable(
        'Isopycnal mixing coefficient', T_GRID, 'm^2/s', 'Isopycnal mixing tensor component',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),
    'K_22': Variable(
        'Isopycnal mixing coefficient', T_GRID, 'm^2/s', 'Isopycnal mixing tensor component',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),
    'K_33': Variable(
        'Isopycnal mixing coefficient', T_GRID, 'm^2/s', 'Isopycnal mixing tensor component',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),
    'Ai_ez': Variable(
        'Isopycnal diffusion coefficient', T_GRID + TENSOR_COMP,
        'Vertical isopycnal diffusion coefficient on eastern face of T cell', '1',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),
    'Ai_nz': Variable(
        'Isopycnal diffusion coefficient', T_GRID + TENSOR_COMP,
        'Vertical isopycnal diffusion coefficient on northern face of T cell', '1',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),
    'Ai_bx': Variable(
        'Isopycnal diffusion coefficient', T_GRID + TENSOR_COMP,
        'Zonal isopycnal diffusion coefficient on bottom face of T cell', '1',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),
    'Ai_by': Variable(
        'Isopycnal diffusion coefficient', T_GRID + TENSOR_COMP,
        'Meridional isopycnal diffusion coefficient on bottom face of T cell', '1',
        active=lambda settings: settings.enable_neutral_diffusion,
    ),

    'B1_gm': Variable(
        'Zonal component of GM streamfunction', V_GRID, 'm^2/s',
        'Zonal component of GM streamfunction',
        active=lambda settings: settings.enable_skew_diffusion,
    ),
    'B2_gm': Variable(
        'Meridional component of GM streamfunction', U_GRID, 'm^2/s',
        'Meridional component of GM streamfunction',
        active=lambda settings: settings.enable_skew_diffusion,
    ),

    'r_bot_var_u': Variable(
        'Bottom friction coeff.', U_HOR, '1/s', 'Zonal bottom friction coefficient',
        active=lambda settings: settings.enable_bottom_friction_var,
    ),
    'r_bot_var_v': Variable(
        'Bottom friction coeff.', V_HOR, '1/s', 'Meridional bottom friction coefficient',
        active=lambda settings: settings.enable_bottom_friction_var,
    ),

    'kappa_gm': Variable(
        'Vertical diffusivity', W_GRID, 'm^2/s', 'Vertical diffusivity',
        active=lambda settings: settings.enable_TEM_friction
    ),

    'tke': Variable(
        'Turbulent kinetic energy', W_GRID + TIMESTEPS, 'm^2/s^2',
        'Turbulent kinetic energy', output=True, write_to_restart=True,
        active=lambda settings: settings.enable_tke,
    ),
    'sqrttke': Variable(
        'Square-root of TKE', W_GRID, 'm/s', 'Square-root of TKE',
        active=lambda settings: settings.enable_tke,
    ),
    'dtke': Variable(
        'Turbulent kinetic energy tendency', W_GRID + TIMESTEPS, 'm^2/s^3',
        'Turbulent kinetic energy tendency', write_to_restart=True,
        active=lambda settings: settings.enable_tke,
    ),
    'Prandtlnumber': Variable(
        'Prandtl number', W_GRID, '', 'Prandtl number',
        active=lambda settings: settings.enable_tke,
    ),
    'mxl': Variable(
        'Mixing length', W_GRID, 'm', 'Mixing length',
        active=lambda settings: settings.enable_tke,
    ),
    'forc_tke_surface': Variable(
        'TKE surface flux', T_HOR, 'm^3/s^3', 'TKE surface flux', output=True,
        active=lambda settings: settings.enable_tke,
    ),
    'tke_diss': Variable(
        'TKE dissipation', W_GRID, 'm^2/s^3', 'TKE dissipation',
        active=lambda settings: settings.enable_tke,
    ),
    'tke_surf_corr': Variable(
        'Correction of TKE surface flux', T_HOR, 'm^3/s^3',
        'Correction of TKE surface flux',
        active=lambda settings: settings.enable_tke,
    ),

    'eke': Variable(
        'meso-scale energy', W_GRID + TIMESTEPS, 'm^2/s^2',
        'meso-scale energy', output=True, write_to_restart=True,
        active=lambda settings: settings.enable_eke,
    ),
    'deke': Variable(
        'meso-scale energy tendency', W_GRID + TIMESTEPS, 'm^2/s^3',
        'meso-scale energy tendency', write_to_restart=True,
        active=lambda settings: settings.enable_eke,
    ),
    'sqrteke': Variable(
        'square-root of eke', W_GRID, 'm/s', 'square-root of eke',
        active=lambda settings: settings.enable_eke,
    ),
    'L_rossby': Variable(
        'Rossby radius', T_HOR, 'm', 'Rossby radius',
        active=lambda settings: settings.enable_eke,
    ),
    'L_rhines': Variable(
        'Rhines scale', W_GRID, 'm', 'Rhines scale',
        active=lambda settings: settings.enable_eke,
    ),
    'eke_len': Variable(
        'Eddy length scale', T_GRID, 'm', 'Eddy length scale',
        active=lambda settings: settings.enable_eke,
    ),
    'eke_diss_iw': Variable(
        'Dissipation of EKE to IW', W_GRID, 'm^2/s^3',
        'Dissipation of EKE to internal waves',
        active=lambda settings: settings.enable_eke,
    ),
    'eke_diss_tke': Variable(
        'Dissipation of EKE to TKE', W_GRID, 'm^2/s^3',
        'Dissipation of EKE to TKE',
        active=lambda settings: settings.enable_eke,
    ),

    'eke_bot_flux': Variable(
        'Flux by bottom friction', T_HOR, 'm^3/s^3', 'Flux by bottom friction',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),
    'eke_topo_hrms': Variable(
        '?', T_HOR, '?', '?',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),
    'eke_topo_lam': Variable(
        '?', T_HOR, '?', '?',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),
    'hrms_k0': Variable(
        '?', T_HOR, '?', '?',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),
    'c_lee': Variable(
        'Lee wave dissipation coefficient', T_HOR, '1/s',
        'Lee wave dissipation coefficient',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),
    'eke_lee_flux': Variable(
        'Lee wave flux', T_HOR, 'm^3/s^3', 'Lee wave flux',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),
    'c_Ri_diss': Variable(
        'Interior dissipation coefficient', W_GRID, '1/s',
        'Interior dissipation coefficient',
        active=lambda settings: settings.enable_eke_leewave_dissipation,
    ),

    'E_iw': Variable(
        'Internal wave energy', W_GRID + TIMESTEPS, 'm^2/s^2',
        'Internal wave energy', output=True, write_to_restart=True,
        active=lambda settings: settings.enable_idemix,
    ),
    'dE_iw': Variable(
        'Internal wave energy tendency', W_GRID + TIMESTEPS, 'm^2/s^2',
        'Internal wave energy tendency', write_to_restart=True,
        active=lambda settings: settings.enable_idemix,
    ),
    'c0': Variable(
        'Vertical IW group velocity', W_GRID, 'm/s',
        'Vertical internal wave group velocity',
        active=lambda settings: settings.enable_idemix,
    ),
    'v0': Variable(
        'Horizontal IW group velocity', W_GRID, 'm/s',
        'Horizontal internal wave group velocity',
        active=lambda settings: settings.enable_idemix,
    ),
    'alpha_c': Variable(
        '?', W_GRID, '?', '?',
        active=lambda settings: settings.enable_idemix,
    ),
    'iw_diss': Variable(
        'IW dissipation', W_GRID, 'm^2/s^3', 'Internal wave dissipation',
        active=lambda settings: settings.enable_idemix,
    ),
    'forc_iw_surface': Variable(
        'IW surface forcing', T_HOR, 'm^3/s^3',
        'Internal wave surface forcing', time_dependent=False, output=True,
        active=lambda settings: settings.enable_idemix,
    ),
    'forc_iw_bottom': Variable(
        'IW bottom forcing', T_HOR, 'm^3/s^3',
        'Internal wave bottom forcing', time_dependent=False, output=True,
        active=lambda settings: settings.enable_idemix,
    ),
}


def manifest_metadata(var_meta, settings):
    """Evaluate callable metadata fields given the current settings."""
    from copy import copy
    out = {}

    for var_name, var_val in var_meta.items():
        var_val = copy(var_val)
        for attr, attr_val in vars(var_val).items():
            if callable(attr_val) and attr != "get_mask":
                setattr(var_val, attr, attr_val(settings))

        out[var_name] = var_val

    return out


def allocate(dimensions, grid, dtype=None, include_ghosts=True, local=True, fill=0):
    from veros.core.operators import numpy as np

    if dtype is None:
        dtype = runtime_settings.float_type

    shape = get_shape(dimensions, grid, include_ghosts=include_ghosts, local=local)
    out = np.full(shape, fill, dtype=dtype)

    if runtime_settings.backend == "numpy":
        out.flags.writeable = False

    return out
