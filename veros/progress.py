import sys
import functools
from time import perf_counter

from veros import logger

try:
    import tqdm
except ImportError:
    has_tqdm = False
else:
    has_tqdm = True

from . import time, logs, runtime_settings as rs, runtime_state as rst

BAR_FORMAT = (
    ' Current iteration: {iteration:<5} ({time:.2f}/{total:.2f}{unit} | {percentage:>4.1f}% | '
    '{rate:.2f}{rate_unit} | {eta:.1f}{eta_unit} left)'
)


class LoggingProgressBar:
    """A simple progress report to logger.info

    Serves as a fallback where TQDM is not available or not feasible (writing to a file,
    in multiprocessing contexts).
    """

    def __init__(self, total, start_time=0, start_iteration=0, time_unit='seconds'):
        self._start_time = start_time
        self._start_iteration = start_iteration
        self._total = total

        _, self._time_unit = time.format_time(total)

    def __enter__(self):
        self._start = perf_counter()
        self._iteration = self._start_iteration
        self._time = self._start_time
        self.flush()
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def advance_time(self, amount, *args, **kwargs):
        self._iteration += 1
        self._time += amount
        self.flush()

    def flush(self):
        report_time = time.convert_time(self._time, 'seconds', self._time_unit)
        total_time = time.convert_time(self._total, 'seconds', self._time_unit)

        if self._time > self._start_time:
            rate_in_seconds = (perf_counter() - self._start) / (self._time - self._start_time)
        else:
            rate_in_seconds = 0
        rate_in_seconds_per_year = rate_in_seconds / time.convert_time(1, 'seconds', 'years')

        rate, rate_unit = time.format_time(rate_in_seconds_per_year)
        eta, eta_unit = time.format_time((self._total - self._time) * rate_in_seconds)

        if self._start_time < self._total:
            percentage = 100 * (self._time - self._start_time) / (self._total - self._start_time)
        else:
            percentage = 100

        logger.info(
            BAR_FORMAT,
            time=report_time,
            total=total_time,
            unit=self._time_unit[0],
            percentage=percentage,
            iteration=self._iteration,
            rate=rate,
            rate_unit='{}/(model year)'.format(rate_unit[0]),
            eta=eta,
            eta_unit=eta_unit[0],
        )


class FancyProgressBar:
    """A fancy progress bar based on TQDM that stays at the bottom of the terminal."""

    def __init__(self, total, start_time=0, start_iteration=0, time_unit='seconds'):
        self._time = self._start_time = start_time
        self._iteration = self._start_iteration = start_iteration
        self._total = total

        total_runlen, time_unit = time.format_time(total)
        self._time_unit = time_unit

        class _VerosTQDM(tqdm.tqdm):
            """Stripped down version of tqdm.tqdm

            We only need TQDM to handle dynamic updates to the progress indicator.
            """
            def __init__(self, *args, **kwargs):
                kwargs.update(leave=True)
                super().__init__(*args, **kwargs)

            @property
            def format_dict(other):
                report_time = time.convert_time(self._time, 'seconds', self._time_unit)
                total_time = time.convert_time(self._total, 'seconds', self._time_unit)
                if self._start_time < self._total:
                    percentage = 100 * (self._time - self._start_time) / (self._total - self._start_time)
                else:
                    percentage = 100

                d = super().format_dict

                if d['elapsed'] > 0:
                    if self._time > self._start_time:
                        rate_in_seconds = d['elapsed'] / (self._time - self._start_time)
                    else:
                        rate_in_seconds = 0
                    rate_in_seconds_per_year = rate_in_seconds / time.convert_time(1, 'seconds', 'years')
                    rate, rate_unit = time.format_time(rate_in_seconds_per_year)
                    eta, eta_unit = time.format_time((self._total - self._time) * rate_in_seconds)
                else:
                    rate, rate_unit = 0, 's'
                    eta, eta_unit = 0, 's'

                d.update(
                    iteration=self._iteration,
                    time=report_time,
                    total=total_time,
                    unit=self._time_unit[0],
                    percentage=percentage,
                    rate=rate,
                    rate_unit='{}/(model year)'.format(rate_unit[0]),
                    eta=eta,
                    eta_unit=eta_unit[0],
                )
                return d

            def format_meter(other, *args, bar_format, **kwargs):
                return bar_format.format(**kwargs)

        self._pbar = _VerosTQDM(
            file=sys.stdout,
            bar_format=BAR_FORMAT
        )

    def __enter__(self, *args, **kwargs):
        self._iteration = self._start_iteration
        self._time = self._start_time
        logs.setup_logging(
            loglevel=rs.loglevel,
            stream_sink=functools.partial(self._pbar.write, file=sys.stdout, end='')
        )
        self._pbar.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        logs.setup_logging(loglevel=rs.loglevel)
        self._pbar.__exit__(*args, **kwargs)

    def advance_time(self, amount):
        self._iteration += 1
        self._time += amount
        self.flush()

    def flush(self):
        self._pbar.refresh()


def get_progress_bar(state, use_tqdm=None):
    if use_tqdm is None:
        use_tqdm = sys.stdout.isatty() and rst.proc_num == 1 and has_tqdm

    if use_tqdm and not has_tqdm:
        raise RuntimeError('tqdm failed to import. Try `pip install tqdm` or set use_tqdm=False.')

    kwargs = dict(
        total=state.settings.runlen + float(state.variables.time),
        start_time=float(state.variables.time),
        start_iteration=int(state.variables.itt)
    )

    if use_tqdm:
        pbar = FancyProgressBar(**kwargs)
    else:
        pbar = LoggingProgressBar(**kwargs)

    return pbar
