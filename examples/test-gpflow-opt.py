import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import GPflow

import sys

sys.path.append('../..')
import opt_tools as ot

X = np.linspace(0, 5, 100)[:, None]
Y = 0.3 * np.sin(2 * X) + 0.05 * rnd.randn(*X.shape)

model = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(1))
model._compile()

optlog = ot.GPflowOptimisationHelper(
    model,
    [
        ot.tasks.DisplayOptimisation(ot.seq_exp_lin(1.0, 1.0)),
        ot.tasks.LogOptimisation(ot.seq_exp_lin(1.0, 1.0), store_fullg=True, store_x=True),
        ot.tasks.StoreOptimisationHistory('./opthist.pkl', ot.seq_exp_lin(1.0, np.inf, 5.0, 5.0), verbose=True)
    ]
)
# optlog = GPflowOptimisationHelper(model, store_x=True,
#                                   disp_sequence=seq_exp_lin(1.0, 1.0),
#                                   hist_sequence=seq_exp_lin(1.0, 1.0),
#                                   store_sequence=seq_exp_lin(1.0, np.inf, 5.0, 5.0),
#                                   store_trigger="iter",
#                                   store_path='./opthist.pkl',
#                                   timeout=0.1)
optlog.callback(model.get_free_state())
try:
    model.optimize(callback=optlog.callback, disp=False)
    optlog.finish(model.get_free_state())
except KeyboardInterrupt:
    finx = optlog.hist.iloc[-1, :].x
    print(finx)

hist = pd.read_pickle('./opthist.pkl')

plt.figure()
plt.subplot(121)
plt.plot(hist.i, hist.f, '-x')
plt.xscale('log')

plt.subplot(122)
plt.plot(hist.i, hist.gnorm, '-x')
plt.xscale('log')
plt.yscale('log')

plt.show()
