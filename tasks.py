import sys
import time
import numpy as np
import pandas as pd


class OptimisationIterationEvent(object):
    def __init__(self, sequence, trigger="iter"):
        self._seq = sequence
        self._trigger = trigger
        self._next = next(self._seq) if self._seq is not None else np.inf

    def event_handler(self, logger, x):
        raise NotImplementedError

    def __call__(self, logger, x, final=False):
        if ((self._trigger == "iter" and logger._i >= self._next) or
                (self._trigger == "time" and time.time() - logger._start_time >= self._next) or
                final):
            self.event_handler(logger, x)
            self._next = next(self._seq)


class DisplayOptimisation(OptimisationIterationEvent):
    def __init__(self, sequence, trigger="iter"):
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._last_disp = (0, time.time())
        print("Iter\tfunc\t\tgrad\t\titer/s\t\tTimestamp")

    def event_handler(self, logger, x):
        # Print and log
        f, g = logger._fg(x)
        iter_per_time = (logger._i - self._last_disp[0]) / (time.time() - self._last_disp[1])
        sys.stdout.write("\r")
        sys.stdout.write(
            "%i\t%e\t%e\t%f\t%s\t" % (logger._i, f, np.linalg.norm(g), iter_per_time, time.ctime()))
        sys.stdout.flush()
        self._last_disp = (logger._i, time.time())


class LogOptimisation(OptimisationIterationEvent):
    def __init__(self, sequence, trigger="iter", hist=None, store_fullg=False, store_x=False):
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._old_hist = hist
        self._store_fullg = store_fullg
        self._store_x = store_x

    def event_handler(self, logger, x, f=None):
        if not hasattr(logger, "hist"):
            if self._old_hist is None:
                logger.hist = pd.DataFrame(columns=['i', 't', 'f', 'gnorm', 'g', 'x'])
            else:
                logger.hist = self._old_hist
                # Todo: Set time and iteration

        if len(logger.hist) > 0 and logger.hist.iloc[-1, :].i == logger._i:
            return
        if f is None:
            f, g = logger._fg(x)
        logger.hist = logger.hist.append(dict(zip(logger.hist.columns,
                                                  (logger._i, time.time() - logger._start_time, f, np.linalg.norm(g),
                                                   g if self._store_fullg else 0.0,
                                                   x.copy() if self._store_x else None))),
                                         ignore_index=True)


class StoreOptimisationHistory(OptimisationIterationEvent):
    def __init__(self, store_path, sequence, trigger="time", verbose=False):
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._store_path = store_path
        self._verbose = verbose

    def event_handler(self, logger, x):
        st = time.time()
        logger.hist.to_pickle(self._store_path)
        if self._verbose:
            print("")
            print("Stored history in %.2fs" % (time.time() - st))
