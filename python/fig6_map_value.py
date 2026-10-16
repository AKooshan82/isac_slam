"""Figure 6: current and persistent value of static-map precision."""
import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# Parameters of the paper's independent one-landmark Gaussian illustration.
map_precision = np.logspace(-4, 4, 401)
epochs = np.arange(11)
Q = np.diag([1.2**2, 1.2**2, np.deg2rad(8.)**2])


def small_map(c):
    """Five-state local information for the one-landmark illustration."""
    p, m = np.array([35., 20.]), np.array([45., 55.])
    d = m-p
    r, u = np.linalg.norm(d), d/np.linalg.norm(d)
    up = np.array([-u[1], u[0]])
    H = np.vstack((np.r_[-u, 0., u], np.r_[-up/r, -1., up/r]))
    F = np.diag([1/.2**2, 1/np.deg2rad(1.5)**2])
    Z = H.T@F@H
    L = np.diag([1/1.5**2, 1/1.5**2, 1/np.deg2rad(10.)**2, c, c])+Z
    return L, Z


def predict_small(L):
    """Frozen transition-information Schur complement, Section IV."""
    Qi = np.linalg.inv(Q)
    A, B, C = L[:3, :3], L[:3, 3:], L[3:, 3:]
    M = A+Qi
    top = np.column_stack((Qi-Qi@np.linalg.solve(M, Qi), Qi@np.linalg.solve(M, B)))
    bottom = np.column_stack((B.T@np.linalg.solve(M, Qi), C-B.T@np.linalg.solve(M, B)))
    out = np.vstack((top, bottom))
    return (out+out.T)/2


# Computation: PEB is a local covariance metric, not nonlinear MAP accuracy.
peb = np.array([isac.metrics(small_map(c)[0])[0] for c in map_precision])
per_decade = -np.gradient(peb, np.log10(map_precision))
L1, Z = small_map(1/9)
L2 = L1.copy()
L2[3:, 3:] += 3*np.eye(2)
persistence = np.empty((len(epochs), 2))
for k in epochs:
    persistence[k] = [isac.metrics(L1)[0], isac.metrics(L2)[0]]
    if k < epochs[-1]:
        L1, L2 = predict_small(L1)+Z, predict_small(L2)+Z
print('Fig. 6a: PEB endpoints (m):', peb[0], peb[-1])
print('Fig. 6b: maximum better-map minus base-map PEB:',
      np.max(persistence[:, 1]-persistence[:, 0]))

# Plotting.
colors = isac.plot_style(diagnostic=True)
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
ax[0].plot(map_precision, peb, color=colors[0], linewidth=1.8, label='Position scale')
right = ax[0].twinx()
right.plot(map_precision, per_decade, color=colors[1], linestyle='--',
           linewidth=1.8, label='Per-decade drop')
ax[0].set(xscale='log', xlim=(1e-4, 1e4), ylim=(1.4, 2.2),
          xlabel='Map prior precision (m$^{-2}$)', ylabel='Local position scale (m)', title='(a)')
right.set(ylim=(0, .4), ylabel='Reduction per decade (m)')
ax[1].plot(epochs, persistence[:, 0], '-o', color=colors[0], linewidth=1.8,
           markersize=2.5, label='Base map')
ax[1].plot(epochs, persistence[:, 1], '--s', color=colors[1], linewidth=1.8,
           markersize=2.5, label='Map precision + 3I')
ax[1].set(xlim=(0, 10), ylim=(1.3, 4.5), xlabel='Prediction / update epoch',
          ylabel='Local position scale (m)', title='(b)')
ax[1].legend(loc='upper left', fontsize=7)
for a in ax:
    a.grid(True)
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig05_map_value'))
