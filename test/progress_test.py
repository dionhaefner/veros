import sys
import re
import time


def test_progress_format(capsys):
    from veros.logs import setup_logging
    setup_logging(stream_sink=sys.stdout)

    from veros.state import VerosState
    from veros.progress import get_progress_bar

    dummy_state = VerosState()
    dummy_state.runlen = 8000
    dummy_state.time = 2000
    dummy_state.itt = 2

    with get_progress_bar(dummy_state, use_tqdm=False) as pbar:
        for _ in range(8):
            time.sleep(0.1)
            pbar.advance_time(1000)

    captured_log = capsys.readouterr()
    assert 'Current iteration:' in captured_log.out

    with get_progress_bar(dummy_state, use_tqdm=True) as pbar:
        for _ in range(8):
            time.sleep(0.1)
            pbar.advance_time(1000)

    captured_tqdm = capsys.readouterr()
    assert 'Current iteration:' in captured_tqdm.out

    def sanitize(prog):
        # remove rates and ETA (inconsistent)
        prog = re.sub(r'\d+\.\d{2}[smh]/\(model year\)', '?s/(model year)', prog)
        prog = re.sub(r'\d+\.\d[smh] left', '? left', prog)
        prog = prog.replace('\r', '\n')
        prog = prog.strip()
        return prog

    def deduplicate(prog):
        # remove repeated identical lines
        out = []
        for line in prog.split('\n'):
            if not out or out[-1] != line:
                out.append(line)
        return '\n'.join(out)

    assert sanitize(captured_log.out) == deduplicate(sanitize(captured_tqdm.out))
