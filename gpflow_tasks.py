"""
Custom opt_tools tasks. Eventually, I think these can be merged into the main package.
"""
import time

import numpy as np

import opt_tools


class GPflowBenchmarkTrackerBase(opt_tools.tasks.GPflowLogOptimisation):
    def __init__(self, test_X, test_Y, sequence, trigger="iter", old_hist=None, store_fullg=False, store_x=True,
                 verbose=False):
        opt_tools.tasks.GPflowLogOptimisation.__init__(self, sequence, trigger, old_hist, store_fullg, store_x)
        self.test_X = test_X
        self.test_Y = test_Y
        self.verbose = verbose


class GPflowBinClassTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowBinClassTracker, self)._get_columns(logger) + ['acc', 'nlpp']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowBinClassTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        p, var = logger.model.predict_y(self.test_X)
        acc = ((p > 0.5).astype('float') == self.test_Y).mean()
        nlpp = np.mean(np.log(p) ** self.test_Y + np.log(1 - p) ** (1 - self.test_Y))
        log_dict.update({'acc': acc, 'nlpp': nlpp})

        if self.verbose:
            print("Benchmarks took %.2fs." % (time.time() - st))

        return log_dict


class GPflowMultiClassificationTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowMultiClassificationTracker, self)._get_columns(logger) + ['acc', 'nlpp']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowMultiClassificationTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        p = np.vstack([logger.model.predict_y(self.test_X[n * 1000:(n + 1) * 1000])[0]
                       for n in range(-(-len(self.test_X // 1000)))])
        assert len(p) == len(self.test_X)
        # acc = ((p > 0.5).astype('float') == self.test_Y).mean()
        acc = (np.argmax(p, 1) == self.test_Y[:, 0]).mean()
        # nlpp = np.mean(np.log(p) ** self.test_Y + np.log(1 - p) ** (1 - self.test_Y))
        nlpp = 0.0
        log_dict.update({'acc': acc, 'nlpp': nlpp})

        if self.verbose:
            print("Benchmarks took %.2fs." % (time.time() - st))

        return log_dict
