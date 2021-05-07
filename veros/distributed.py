import functools

from . import runtime_settings as rs, runtime_state as rst
from .routines import CURRENT_CONTEXT

SCATTERED_DIMENSIONS = (
    ('xt', 'xu'),
    ('yt', 'yu')
)


def dist_context_only(function=None, *, noop_return_arg=None):
    def decorator(function):
        @functools.wraps(function)
        def dist_context_only_wrapper(*args, **kwargs):
            if rst.proc_num == 1 or not CURRENT_CONTEXT.is_dist_safe:
                # no-op for sequential execution
                if noop_return_arg is None:
                    return None

                # return input array unchanged
                return args[noop_return_arg]

            return function(*args, **kwargs)

        return dist_context_only_wrapper

    if function is not None:
        return decorator(function)

    return decorator


def send(buf, dest, comm, tag=None, token=None):
    kwargs = {}
    if tag is not None:
        kwargs.update(tag=tag)

    if rs.backend == 'jax':
        from mpi4jax import send
        return send(buf, dest=dest, comm=comm, token=token, **kwargs)

    comm.Send(ascontiguousarray(buf), dest=dest, **kwargs)
    return None


def recv(buf, source, comm, tag=None, token=None):
    kwargs = {}
    if tag is not None:
        kwargs.update(tag=tag)

    if rs.backend == 'jax':
        from mpi4jax import recv
        return recv(buf, source=source, comm=comm, token=token, **kwargs)

    buf = buf.copy()
    comm.Recv(buf, source=source, **kwargs)
    return buf, None


def sendrecv(sendbuf, recvbuf, source, dest, comm, sendtag=None, recvtag=None, token=None):
    kwargs = {}

    if sendtag is not None:
        kwargs.update(sendtag=sendtag)

    if recvtag is not None:
        kwargs.update(recvtag=recvtag)

    if rs.backend == 'jax':
        from mpi4jax import sendrecv
        return sendrecv(sendbuf, recvbuf, source=source, dest=dest, comm=comm, token=token, **kwargs)

    recvbuf = recvbuf.copy()
    comm.Sendrecv(
        sendbuf=ascontiguousarray(sendbuf), recvbuf=recvbuf,
        source=source, dest=dest,
        **kwargs
    )
    return recvbuf, None


def bcast(buf, comm, root=0, token=None):
    if rs.backend == 'jax':
        from mpi4jax import bcast
        return bcast(buf, root=root, comm=comm, token=token)

    return comm.bcast(buf, root=root), None


def allreduce(buf, op, comm, token=None):
    if rs.backend == 'jax':
        from mpi4jax import allreduce
        return allreduce(buf, op=op, comm=comm, token=token)

    from veros.core.operators import numpy as np
    recvbuf = np.empty_like(buf)
    comm.Allreduce(ascontiguousarray(buf), recvbuf, op=op)
    return recvbuf, None


def ascontiguousarray(arr):
    assert rs.backend == 'numpy'
    import numpy
    return numpy.ascontiguousarray(arr)


def validate_decomposition(dimensions):
    nx, ny = dimensions["xt"], dimensions["yt"]

    if rs.mpi_comm is None:
        if (rs.num_proc[0] > 1 or rs.num_proc[1] > 1):
            raise RuntimeError('mpi4py is required for distributed execution')
        return

    comm_size = rs.mpi_comm.Get_size()
    proc_num = rs.num_proc[0] * rs.num_proc[1]
    if proc_num != comm_size:
        raise RuntimeError('number of processes ({}) does not match size of communicator ({})'
                           .format(proc_num, comm_size))

    if nx % rs.num_proc[0]:
        raise ValueError('processes do not divide domain evenly in x-direction')

    if ny % rs.num_proc[1]:
        raise ValueError('processes do not divide domain evenly in y-direction')


def get_chunk_size(nx, ny):
    return (nx // rs.num_proc[0], ny // rs.num_proc[1])


def proc_rank_to_index(rank):
    return (rank % rs.num_proc[0], rank // rs.num_proc[0])


def proc_index_to_rank(ix, iy):
    return ix + iy * rs.num_proc[0]


def get_chunk_slices(nx, ny, dim_grid, proc_idx=None, include_overlap=False):
    if proc_idx is None:
        proc_idx = proc_rank_to_index(rst.proc_rank)

    px, py = proc_idx
    nxl, nyl = get_chunk_size(nx, ny)

    if include_overlap:
        sxl = 0 if px == 0 else 2
        sxu = nxl + 4 if (px + 1) == rs.num_proc[0] else nxl + 2
        syl = 0 if py == 0 else 2
        syu = nyl + 4 if (py + 1) == rs.num_proc[1] else nyl + 2
    else:
        sxl = syl = 0
        sxu = nxl
        syu = nyl

    global_slice, local_slice = [], []

    for dim in dim_grid:
        if dim in SCATTERED_DIMENSIONS[0]:
            global_slice.append(slice(sxl + px * nxl, sxu + px * nxl))
            local_slice.append(slice(sxl, sxu))
        elif dim in SCATTERED_DIMENSIONS[1]:
            global_slice.append(slice(syl + py * nyl, syu + py * nyl))
            local_slice.append(slice(syl, syu))
        else:
            global_slice.append(slice(None))
            local_slice.append(slice(None))

    return tuple(global_slice), tuple(local_slice)


def get_process_neighbors(cyclic=False):
    this_x, this_y = proc_rank_to_index(rst.proc_rank)

    if this_x == 0:
        if cyclic:
            west = rs.num_proc[0] - 1
        else:
            west = None
    else:
        west = this_x - 1

    if this_x == rs.num_proc[0] - 1:
        if cyclic:
            east = 0
        else:
            east = None
    else:
        east = this_x + 1

    south = this_y - 1 if this_y != 0 else None
    north = this_y + 1 if this_y != (rs.num_proc[1] - 1) else None

    neighbors = dict(
        # direct neighbors
        west=(west, this_y),
        south=(this_x, south),
        east=(east, this_y),
        north=(this_x, north),
        # corners
        southwest=(west, south),
        southeast=(east, south),
        northeast=(east, north),
        northwest=(west, north),
    )

    global_neighbors = {
        k: proc_index_to_rank(*i) if None not in i else None for k, i in neighbors.items()
    }
    return global_neighbors


@dist_context_only
def exchange_overlap(arr, var_grid, cyclic):
    from veros.core.operators import numpy as np, update, at

    # start west, go clockwise
    send_order = (
        'west',
        'northwest',
        'north',
        'northeast',
        'east',
        'southeast',
        'south',
        'southwest',
    )

    # start east, go clockwise
    recv_order = (
        'east',
        'southeast',
        'south',
        'southwest',
        'west',
        'northwest',
        'north',
        'northeast',
    )

    if len(var_grid) < 2:
        d1, d2 = var_grid[0], None
    else:
        d1, d2 = var_grid[:2]

    if d1 not in SCATTERED_DIMENSIONS[0] and d1 not in SCATTERED_DIMENSIONS[1] and d2 not in SCATTERED_DIMENSIONS[1]:
        # neither x nor y dependent, nothing to do
        return arr

    proc_neighbors = get_process_neighbors(cyclic)

    if d1 in SCATTERED_DIMENSIONS[0] and d2 in SCATTERED_DIMENSIONS[1]:
        overlap_slices_from = dict(
            west=(slice(2, 4), slice(0, None), Ellipsis),
            south=(slice(0, None), slice(2, 4), Ellipsis),
            east=(slice(-4, -2), slice(0, None), Ellipsis),
            north=(slice(0, None), slice(-4, -2), Ellipsis),
            southwest=(slice(2, 4), slice(2, 4), Ellipsis),
            southeast=(slice(-4, -2), slice(2, 4), Ellipsis),
            northeast=(slice(-4, -2), slice(-4, -2), Ellipsis),
            northwest=(slice(2, 4), slice(-4, -2), Ellipsis),
        )

        overlap_slices_to = dict(
            west=(slice(0, 2), slice(0, None), Ellipsis),
            south=(slice(0, None), slice(0, 2), Ellipsis),
            east=(slice(-2, None), slice(0, None), Ellipsis),
            north=(slice(0, None), slice(-2, None), Ellipsis),
            southwest=(slice(0, 2), slice(0, 2), Ellipsis),
            southeast=(slice(-2, None), slice(0, 2), Ellipsis),
            northeast=(slice(-2, None), slice(-2, None), Ellipsis),
            northwest=(slice(0, 2), slice(-2, None), Ellipsis),
        )

    else:
        if d1 in SCATTERED_DIMENSIONS[0]:
            send_order = ('west', 'east')
            recv_order = ('east', 'west')
        elif d1 in SCATTERED_DIMENSIONS[1]:
            send_order = ('north', 'south')
            recv_order = ('south', 'north')
        else:
            raise NotImplementedError()

        overlap_slices_from = dict(
            west=(slice(2, 4), Ellipsis),
            south=(slice(2, 4), Ellipsis),
            east=(slice(-4, -2), Ellipsis),
            north=(slice(-4, -2), Ellipsis),
        )

        overlap_slices_to = dict(
            west=(slice(0, 2), Ellipsis),
            south=(slice(0, 2), Ellipsis),
            east=(slice(-2, None), Ellipsis),
            north=(slice(-2, None), Ellipsis),
        )

    token = None

    for send_dir, recv_dir in zip(send_order, recv_order):
        send_proc = proc_neighbors[send_dir]
        recv_proc = proc_neighbors[recv_dir]

        if send_proc is None and recv_proc is None:
            continue

        recv_idx = overlap_slices_to[recv_dir]
        recv_arr = np.empty_like(arr[recv_idx])

        send_idx = overlap_slices_from[send_dir]
        send_arr = arr[send_idx]

        if send_proc is None:
            recv_arr, token = recv(recv_arr, recv_proc, rs.mpi_comm, token=token)
            arr = update(arr, at[recv_idx], recv_arr)
        elif recv_proc is None:
            token = send(send_arr, send_proc, rs.mpi_comm, token=token)
        else:
            recv_arr, token = sendrecv(
                send_arr, recv_arr,
                source=recv_proc, dest=send_proc,
                comm=rs.mpi_comm, token=token
            )
            arr = update(arr, at[recv_idx], recv_arr)

    return arr


def _memoize(function):
    cached = {}

    @functools.wraps(function)
    def memoized(*args):
        from mpi4py import MPI

        # MPI Comms are not hashable, so we use the underlying handle instead
        cache_args = tuple(
            MPI._handleof(arg) if isinstance(arg, MPI.Comm)
            else arg for arg in args
        )

        if cache_args not in cached:
            cached[cache_args] = function(*args)

        return cached[cache_args]

    return memoized


@_memoize
def _mpi_comm_along_axis(comm, procs, rank):
    return comm.Split(procs, rank)


@dist_context_only
def _reduce(arr, op, axis=None):
    from veros.core.operators import numpy as np

    if axis is None:
        comm = rs.mpi_comm
    else:
        assert axis in (0, 1)
        pi = proc_rank_to_index(rst.proc_rank)
        other_axis = 1 - axis
        comm = _mpi_comm_along_axis(rs.mpi_comm, pi[other_axis], rst.proc_rank)

    if np.isscalar(arr):
        squeeze = True
        arr = np.array([arr])
    else:
        squeeze = False

    res, _ = allreduce(arr, op=op, comm=comm)

    if squeeze:
        res = res[0]

    return res


@dist_context_only(noop_return_arg=0)
def global_and(arr, axis=None):
    from mpi4py import MPI
    return _reduce(arr, MPI.LAND, axis=axis)


@dist_context_only(noop_return_arg=0)
def global_or(arr, axis=None):
    from mpi4py import MPI
    return _reduce(arr, MPI.LOR, axis=axis)


@dist_context_only(noop_return_arg=0)
def global_max(arr, axis=None):
    from mpi4py import MPI
    return _reduce(arr, MPI.MAX, axis=axis)


@dist_context_only(noop_return_arg=0)
def global_min(arr, axis=None):
    from mpi4py import MPI
    return _reduce(arr, MPI.MIN, axis=axis)


@dist_context_only(noop_return_arg=0)
def global_sum(arr, axis=None):
    from mpi4py import MPI
    return _reduce(arr, MPI.SUM, axis=axis)


@dist_context_only(noop_return_arg=2)
def _gather_1d(nx, ny, arr, dim):
    from veros.core.operators import numpy as np, update, at

    assert dim in (0, 1)

    otherdim = 1 - dim
    pi = proc_rank_to_index(rst.proc_rank)
    if pi[otherdim] != 0:
        return arr

    dim_grid = ['xt' if dim == 0 else 'yt'] + [None] * (arr.ndim - 1)
    gidx, idx = get_chunk_slices(nx, ny, dim_grid, include_overlap=True)
    sendbuf = arr[idx]

    token = None

    if rst.proc_rank == 0:
        buffer_list = []
        for proc in range(1, rst.proc_num):
            pi = proc_rank_to_index(proc)
            if pi[otherdim] != 0:
                continue
            idx_g, idx_l = get_chunk_slices(nx, ny, dim_grid, include_overlap=True, proc_idx=pi)
            recvbuf = np.empty_like(arr[idx_l])
            recvbuf, token = recv(recvbuf, source=proc, tag=20, comm=rs.mpi_comm, token=token)
            buffer_list.append((idx_g, recvbuf))

        out_shape = ((nx + 4, ny + 4)[dim],) + arr.shape[1:]
        out = np.empty(out_shape, dtype=arr.dtype)
        out = update(out, at[gidx], sendbuf)

        for idx, val in buffer_list:
            out = update(out, at[idx], val)

        return out

    else:
        send(sendbuf, dest=0, tag=20, comm=rs.mpi_comm)
        return arr


@dist_context_only(noop_return_arg=2)
def _gather_xy(nx, ny, arr):
    from veros.core.operators import numpy as np, update, at

    nxi, nyi = get_chunk_size(nx, ny)
    assert arr.shape[:2] == (nxi + 4, nyi + 4), arr.shape

    dim_grid = ['xt', 'yt'] + [None] * (arr.ndim - 2)
    gidx, idx = get_chunk_slices(nx, ny, dim_grid, include_overlap=True)
    sendbuf = arr[idx]

    token = None

    if rst.proc_rank == 0:
        buffer_list = []
        for proc in range(1, rst.proc_num):
            idx_g, idx_l = get_chunk_slices(
                nx, ny, dim_grid, include_overlap=True,
                proc_idx=proc_rank_to_index(proc)
            )
            recvbuf = np.empty_like(arr[idx_l])
            recvbuf, token = recv(recvbuf, source=proc, tag=30, comm=rs.mpi_comm, token=token)
            buffer_list.append((idx_g, recvbuf))

        out_shape = (nx + 4, ny + 4) + arr.shape[2:]
        out = np.empty(out_shape, dtype=arr.dtype)
        out = update(out, at[gidx], sendbuf)

        for idx, val in buffer_list:
            out = update(out, at[idx], val)

        return out

    send(sendbuf, dest=0, tag=30, comm=rs.mpi_comm)
    return arr


@dist_context_only(noop_return_arg=0)
def gather(arr, dimensions, var_grid):
    nx, ny = dimensions["xt"], dimensions["yt"]

    if len(var_grid) < 2:
        d1, d2 = var_grid[0], None
    else:
        d1, d2 = var_grid[:2]

    if d1 not in SCATTERED_DIMENSIONS[0] and d1 not in SCATTERED_DIMENSIONS[1] and d2 not in SCATTERED_DIMENSIONS[1]:
        # neither x nor y dependent, nothing to do
        return arr

    if d1 in SCATTERED_DIMENSIONS[0] and d2 not in SCATTERED_DIMENSIONS[1]:
        # only x dependent
        return _gather_1d(nx, ny, arr, 0)

    elif d1 in SCATTERED_DIMENSIONS[1]:
        # only y dependent
        return _gather_1d(nx, ny, arr, 1)

    elif d1 in SCATTERED_DIMENSIONS[0] and d2 in SCATTERED_DIMENSIONS[1]:
        # x and y dependent
        return _gather_xy(nx, ny, arr)

    else:
        raise NotImplementedError()


@dist_context_only(noop_return_arg=0)
def _scatter_constant(arr):
    return bcast(arr, rs.mpi_comm, root=0)


@dist_context_only(noop_return_arg=2)
def _scatter_1d(nx, ny, arr, dim):
    from veros.core.operators import numpy as np, update, at

    assert dim in (0, 1)

    out_nx = get_chunk_size(nx, ny)[dim]
    dim_grid = ['xt' if dim == 0 else 'yt'] + [None] * (arr.ndim - 1)
    _, local_slice = get_chunk_slices(nx, ny, dim_grid, include_overlap=True)

    token = None

    if rst.proc_rank == 0:
        recvbuf = arr[local_slice]

        for proc in range(1, rst.proc_num):
            global_slice, _ = get_chunk_slices(
                nx, ny, dim_grid, include_overlap=True, proc_idx=proc_rank_to_index(proc)
            )
            sendbuf = arr[global_slice]
            token = send(sendbuf, dest=proc, tag=40, comm=rs.mpi_comm, token=token)

        # arr changes shape in main process
        arr = np.zeros((out_nx + 4,) + arr.shape[1:], dtype=arr.dtype)
    else:
        recvbuf, _ = recv(arr[local_slice], source=0, tag=40, comm=rs.mpi_comm)

    arr = update(arr, at[local_slice], recvbuf)
    arr = exchange_overlap(arr, ['xt' if dim == 0 else 'yt'], cyclic=False)

    return arr


@dist_context_only(noop_return_arg=2)
def _scatter_xy(nx, ny, arr):
    from veros.core.operators import numpy as np, update, at

    nxi, nyi = get_chunk_size(nx, ny)

    dim_grid = ['xt', 'yt'] + [None] * (arr.ndim - 2)
    _, local_slice = get_chunk_slices(nx, ny, dim_grid, include_overlap=True)

    token = None

    if rst.proc_rank == 0:
        recvbuf = arr[local_slice]

        for proc in range(1, rst.proc_num):
            global_slice, _ = get_chunk_slices(
                nx, ny, dim_grid, include_overlap=True, proc_idx=proc_rank_to_index(proc)
            )
            sendbuf = arr[global_slice]
            token = send(sendbuf, dest=proc, tag=50, comm=rs.mpi_comm, token=token)

        # arr changes shape in main process
        arr = np.empty((nxi + 4, nyi + 4) + arr.shape[2:], dtype=arr.dtype)
    else:
        recvbuf = np.empty_like(arr[local_slice])
        recvbuf, _ = recv(recvbuf, source=0, tag=50, comm=rs.mpi_comm)

    arr = update(arr, at[local_slice], recvbuf)
    arr = exchange_overlap(arr, ['xt', 'yt'], cyclic=False)

    return arr


@dist_context_only(noop_return_arg=0)
def scatter(arr, dimensions, var_grid):
    from veros.core.operators import numpy as np
    nx, ny = dimensions["xt"], dimensions["yt"]

    if len(var_grid) < 2:
        d1, d2 = var_grid[0], None
    else:
        d1, d2 = var_grid[:2]

    arr = np.asarray(arr)

    if d1 not in SCATTERED_DIMENSIONS[0] and d1 not in SCATTERED_DIMENSIONS[1] and d2 not in SCATTERED_DIMENSIONS[1]:
        # neither x nor y dependent
        return _scatter_constant(arr)

    if d1 in SCATTERED_DIMENSIONS[0] and d2 not in SCATTERED_DIMENSIONS[1]:
        # only x dependent
        return _scatter_1d(nx, ny, arr, 0)

    elif d1 in SCATTERED_DIMENSIONS[1]:
        # only y dependent
        return _scatter_1d(nx, ny, arr, 1)

    elif d1 in SCATTERED_DIMENSIONS[0] and d2 in SCATTERED_DIMENSIONS[1]:
        # x and y dependent
        return _scatter_xy(nx, ny, arr)

    else:
        raise NotImplementedError()


@dist_context_only
def barrier():
    rs.mpi_comm.barrier()


@dist_context_only
def abort():
    rs.mpi_comm.Abort()
