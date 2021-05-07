import threading
import contextlib

from veros import logger

from ... import runtime_settings, runtime_state


@contextlib.contextmanager
def threaded_io(filepath, mode):
    """
    If using IO threads, start a new thread to write the HDF5 data to disk.
    """
    import h5py
    if runtime_settings.use_io_threads:
        _wait_for_disk(filepath)
        _io_locks[filepath].clear()
    kwargs = {}
    if runtime_state.proc_num > 1:
        kwargs.update(
            driver='mpio',
            comm=runtime_settings.mpi_comm
        )
    h5file = h5py.File(filepath, mode, **kwargs)
    try:
        yield h5file
    finally:
        if runtime_settings.use_io_threads:
            threading.Thread(target=_write_to_disk, args=(h5file, filepath)).start()
        else:
            _write_to_disk(h5file, filepath)


_io_locks = {}


def _add_to_locks(file_id):
    """
    If there is no lock for file_id, create one
    """
    if file_id not in _io_locks:
        _io_locks[file_id] = threading.Event()
        _io_locks[file_id].set()


def _wait_for_disk(file_id):
    """
    Wait for the lock of file_id to be released
    """
    logger.debug('Waiting for lock {} to be released'.format(file_id))
    _add_to_locks(file_id)
    lock_released = _io_locks[file_id].wait(runtime_settings.io_timeout)
    if not lock_released:
        raise RuntimeError('Timeout while waiting for disk IO to finish')


def _write_to_disk(h5file, file_id):
    """
    Sync HDF5 data to disk, close file handle, and release lock.
    May run in a separate thread.
    """
    try:
        h5file.close()
    finally:
        if runtime_settings.use_io_threads and file_id is not None:
            _io_locks[file_id].set()
