import time


class OptimisationHelper(object):
    def __init__(self, f, tasks, g=None, chaincallback=None):
        self._f = f
        self._g = g
        self.tasks = tasks
        self._chaincallback = chaincallback
        self._i = 0
        self._start_time = time.time()

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
        self._i += 1
        for task in self.tasks:
            task(self, x, final=final)

        if self._chaincallback is not None:
            self._chaincallback(x)

    def finish(self, x):
        for task in self.tasks:
            task.event_handler(self, x)


class GPflowOptimisationHelper(OptimisationHelper):
    def __init__(self, model, tasks, chaincallback=None):
        OptimisationHelper.__init__(self, None, tasks, None, chaincallback)
        self.model = model

    def _fg(self, x):
        return self.model._objective(x)


def seq_exp_lin(growth, max, start=1.0, start_jump=None):
    start_jump = start if start_jump is None else start_jump
    gap = start_jump
    last = start - start_jump
    while 1:
        yield gap + last
        last = last + gap
        gap = min(gap * growth, max)
