import itertools
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
            while self._next < (logger._i if self._trigger == "iter" else time.time() - logger._start_time):
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
    hist_name = "hist"

    def __init__(self, sequence, trigger="iter", hist=None, store_fullg=False, store_x=False):
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._old_hist = hist
        self._store_fullg = store_fullg
        self._store_x = store_x

    def _get_hist(self, logger):
        return getattr(logger, self.hist_name)

    def _set_hist(self, logger, hist):
        return setattr(logger, self.hist_name, hist)

    def _get_columns(self, logger):
        return ['i', 't', 'f', 'gnorm', 'g', 'x']

    def _setup_logger(self, logger):
        if self._old_hist is None:
            self._set_hist(logger, pd.DataFrame(columns=self._get_columns(logger)))
        else:
            self._set_hist(logger, self._old_hist)
            hist = self._get_hist(logger)
            logger._i = hist.i.max()
            logger._start_time = time.time() - hist.t.max()

    def _get_record(self, logger, x, f=None):
        if f is None:
            f, g = logger._fg(x)
        return dict(zip(
            self._get_hist(logger).columns,
            (logger._i, time.time() - logger._start_time, f, np.linalg.norm(g), g if self._store_fullg else 0.0,
             x.copy() if self._store_x else None)
        ))

    def event_handler(self, logger, x, f=None):
        if not hasattr(logger, self.hist_name):
            self._setup_logger(logger)

        hist = self._get_hist(logger)
        if len(hist) > 0 and hist.iloc[-1, :].i == logger._i:
            return

        self._set_hist(logger, hist.append(self._get_record(logger, x), ignore_index=True))


class GPflowLogOptimisation(LogOptimisation):
    def _get_columns(self, logger):
        return ['i', 't', 'f', 'gnorm', 'g'] + list(logger.model.get_parameter_dict().keys())

    def _get_record(self, logger, x, f=None):
        if f is None:
            f, g = logger._fg(x)
        log_dict = dict(zip(
            self._get_hist(logger).columns[:5],
            (logger._i, time.time() - logger._start_time, f, np.linalg.norm(g), g if self._store_fullg else 0.0)
        ))
        if self._store_x:
            log_dict.update(logger.model.get_samples_df(x[None, :].copy()).iloc[0, :].to_dict())
        return log_dict


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


class Timeout(OptimisationIterationEvent):
    def __init__(self, threshold, trigger="time"):
        OptimisationIterationEvent.__init__(self, itertools.cycle([threshold]), trigger)
        self._triggered = False

    def event_handler(self, logger, x):
        if not self._triggered:
            self._triggered = True
            logger.finish(x)
