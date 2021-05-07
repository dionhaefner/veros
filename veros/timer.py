import timeit


class Timer:
    def __init__(self, inactive=False):
        self.total_time = 0
        self.last_time = 0
        self.active = not inactive

    def __enter__(self):
        self.start_time = timeit.default_timer()

    def __exit__(self, type, value, traceback):
        self.last_time = timeit.default_timer() - self.start_time

        if self.active:
            self.total_time += self.last_time

    def get_time(self):
        return self.total_time

    def get_last_time(self):
        return self.last_time
