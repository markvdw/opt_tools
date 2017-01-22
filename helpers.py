import contextlib
import time

import numpy as np


class Stopwatch(object):
    def __init__(self, elapsed_time=0.0):
        self._start_time = None
        self._elapsed_time = elapsed_time

    def start(self):
        if self._start_time is not None:
            raise RuntimeError("Can not start stopwatch unless it is stopped.")
        self._start_time = time.time()

    def stop(self):
        if self._start_time is None:
            raise RuntimeError("Can not stop stopwatch unless it is running.")
        self._elapsed_time += time.time() - self._start_time
        self._start_time = None

    def add_time(self, time):
        self._elapsed_time += time

    @property
    def running(self):
        return self._start_time is not None

    @property
    def elapsed_time(self):
        if self.running:
            return self._elapsed_time + time.time() - self._start_time
        else:
            return self._elapsed_time

    @contextlib.contextmanager
    def pause(self):
        self.stop()
        yield
        self.start()


class OptimisationHelper(object):
    def __init__(self, f, tasks, g=None, chaincallback=None):
        self._f = f
        self._g = g
        self.tasks = tasks
        self._chaincallback = chaincallback
        self._i = 0
        self._opt_timer = Stopwatch()
        self._opt_timer.start()
        self._total_timer = Stopwatch()
        self._total_timer.start()

        for task in self.tasks:
            task.setup(self)

    def _fg(self, x):
        """
        Distinguish between separate functions for f and g or a single one, and call the appropriate ones.
        :param x:
        :return:
        """
        f = self._f(x)
        if type(f) is tuple or type(f) is list:
            return f
        elif self._g is not None:
            return f, self._g(x)
        else:
            return f, 0.0

    def callback(self, x, final=False):
        with self._opt_timer.pause():
            self._i += 1
            for task in self.tasks:
                task(self, x, final=final)

            if self._chaincallback is not None:
                self._chaincallback(x)

    def finish(self, x):
        for task in self.tasks:
            task(self, x, final=True)


class GPflowOptimisationHelper(OptimisationHelper):
    def __init__(self, model, tasks, chaincallback=None):
        self.model = model
        OptimisationHelper.__init__(self, None, tasks, None, chaincallback)

        # Variables for `_fg` memoisation.
        self._prev_x = None
        self._prev_val = None

    def _fg(self, x):
        if self._prev_x is None or np.any(self._prev_x != x):
            self._prev_x = x.copy()
            self._prev_val = self.model._objective(x)
            self.model.num_fevals -= 1
        return self._prev_val


def seq_exp_lin(growth, max, start=1.0, start_jump=None):
    start_jump = start if start_jump is None else start_jump
    gap = start_jump
    last = start - start_jump
    while 1:
        yield gap + last
        last = last + gap
        gap = min(gap * growth, max)
