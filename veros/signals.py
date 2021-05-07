import signal
import contextlib
import functools

from veros import logger


def do_not_disturb(function):
    """Decorator that catches SIGINT and SIGTERM signals (e.g. after keyboard interrupt)
    and makes sure that the function body is executed before exiting.

    Useful for ensuring that output files are written properly.
    """
    signals = (signal.SIGINT, signal.SIGTERM)

    @functools.wraps(function)
    def dnd_wrapper(*args, **kwargs):
        old_handlers = {s: signal.getsignal(s) for s in signals}
        signal_received = {'sig': None, 'frame': None}

        def handler(sig, frame):
            if signal_received['sig'] is None:
                signal_received['sig'] = sig
                signal_received['frame'] = frame
                logger.error(f'Signal {sig} received - cleaning up before exit (interrupt again to force exit)')
            else:
                # force quit if more than one signal is received
                old_handlers[sig](sig, frame)

        for s in signals:
            signal.signal(s, handler)

        try:
            res = function(*args, **kwargs)

        finally:
            for s in signals:
                signal.signal(s, old_handlers[s])
            sig = signal_received['sig']
            if sig is not None:
                old_handlers[sig](signal_received['sig'], signal_received['frame'])

        return res

    return dnd_wrapper


@contextlib.contextmanager
def signals_to_exception(signals=(signal.SIGINT, signal.SIGTERM)):
    """Context manager that converts system signals to exceptions.

    This allows for a graceful exit after receiving SIGTERM (e.g. through
    `kill` on UNIX systems).

    Example:
       >>> with signals_to_exception():
       >>>     try:
       >>>         # do something
       >>>     except SystemExit:
       >>>         # graceful exit even upon receiving interrupt signal
    """
    def signal_to_exception(sig, frame):
        logger.critical('Received interrupt signal {}', sig)
        raise SystemExit('Aborted')

    old_signals = {}
    for s in signals:
        # override signals with our handler
        old_signals[s] = signal.getsignal(s)
        signal.signal(s, signal_to_exception)

    try:
        yield

    finally:
        # re-attach old signals
        for s in signals:
            signal.signal(s, old_signals[s])
