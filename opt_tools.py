import sys
import time
import numpy as np
import pandas as pd


class OptimisationLogger(object):
    def __init__(self, f, disp_sequence=None, hist_sequence=None, g=None, chaincallback=None, store_fullg=False,
                 store_x=False):
        # Options
        self._f = f
        self._g = g
        self._chaincallback = chaincallback
        self._disp_seq = disp_sequence if disp_sequence is not None else seq_exp_lin(1.5, 1000)
        self._hist_seq = hist_sequence if disp_sequence is not None else seq_exp_lin(1.5, 1000)
        self._store_fullg = store_fullg
        self._store_x = store_x

        # Working variables
        self._next_disp = next(self._disp_seq)
        self._last_disp = (0, time.time())
        self._next_hist = next(self._hist_seq) if self._hist_seq is not None else np.inf
        self._i = 0
        self._start_time = time.time()

        # Public members
        self.hist = pd.DataFrame(columns=['i', 't', 'f', 'gnorm', 'g', 'x'])

        print("Iter\tfunc\t\tgrad\t\titer/s\t\tTimestamp")

    def _fg(self, x):
        f = self._f(x)
        if type(f) is tuple or type(f) is list:
            return f
        elif self._g is not None:
            g = self._g(x)
            return f, g
        else:
            return f, 0.0

    def callback(self, x):
        self._i += 1
        f = None
        if self._i >= self._next_disp:
            # Print and log
            f, g = self._fg(x)
            time_per_iter = (self._i - self._last_disp[0]) / (time.time() - self._last_disp[1])
            sys.stdout.write("\r")
            sys.stdout.write(
                "%i\t%e\t%e\t%f\t%s\t" % (self._i, f, np.linalg.norm(g), time_per_iter, time.ctime()))
            sys.stdout.flush()
            self._last_disp = (self._i, time.time())
            self._next_disp = next(self._disp_seq)
        if self._i >= self._next_hist:
            if f is None:
                f, g = self._fg(x)
            self.hist = self.hist.append(dict(zip(self.hist.columns,
                                                  (self._i, time.time() - self._start_time, f, np.linalg.norm(g),
                                                   g if self._store_fullg else 0.0,
                                                   x.copy() if self._store_x else None))),
                                         ignore_index=True)
            self._next_hist = next(self._hist_seq)

        if self._chaincallback is not None:
            self._chaincallback(x)


class OptimisationHelper(OptimisationLogger):
    def __init__(self, f, disp_sequence=None, hist_sequence=None, store_sequence=None, g=None, chaincallback=None,
                 store_fullg=False, store_x=False, timeout=np.inf, store_path=None):
        """
        :param f: Returns objective function value or list/tuple of objfunc + gradient
        :param disp_sequence: Sequence of iterations when to display to screen
        :param hist_sequence: Sequence of iterations when to record in history
        :param store_sequence: Sequence of times when to store history to disk
        :param g:
        :param chaincallback: Not implemented yet
        :param store_fullg: Keep history of the full gradient evaluation
        :param store_x:
        :param timeout:
        :param store_path: File where history is stored.
        """
        super(OptimisationHelper, self).__init__(f, disp_sequence, hist_sequence, g, chaincallback, store_fullg,
                                                 store_x)
        self.timeout = timeout
        self._store_seq = store_sequence
        self._next_store = next(self._store_seq) if self._store_seq is not None else np.inf
        self._store_path = store_path
        if store_sequence is not None and store_path is None:
            raise ValueError("Need a `store_path` to store history file.")

    def callback(self, x):
        if (time.time() - self._start_time) > self.timeout:
            raise KeyboardInterrupt

        super(OptimisationHelper, self).callback(x)
        if time.time() - self._start_time > self._next_store:
            self.hist.to_pickle(self._store_path)
            self._next_store = next(self._store_seq)


def seq_exp_lin(growth, max_gap, start=1.0, start_jump=1.0):
    gap = start_jump
    last = start - start_jump
    while 1:
        yield gap + last
        last = last + gap
        gap = min(gap * growth, max_gap)

every_iter = seq_exp_lin(1.0, 1.0)
