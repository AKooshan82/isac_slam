"""Figure 3: unknown-gain range information and the fixed-BS gauge."""
import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# Parameters from the corrected diagnostic experiment.
cfg = isac.defaults()
B = np.logspace(5, 9, 401)
Gamma = 100.
N = 128

# Computation: conditional CRBs and scaled full-model eigenvalues.
f = (np.arange(N)-N/2)[None, :]*B[:, None]/N
variance = np.mean((f-f.mean(axis=1, keepdims=True))**2, axis=1)
known = cfg['speed']/2/np.sqrt(8*np.pi**2*Gamma*np.mean((cfg['fc']+f)**2, axis=1))
unknown = cfg['speed']/2/np.sqrt(8*np.pi**2*Gamma*variance)
w = isac.beams(cfg['q0'], cfg)
Z, _, meta = isac.efim_atoms(cfg['q0'], w, cfg)
e = np.full(cfg['R'], cfg['Emax']/cfg['R'])
S = np.diag(cfg['state_scale'])
env = np.einsum('r,ijr->ij', e, meta['Zsense'])
total = isac.post(np.zeros_like(cfg['Lambda0']), Z, e)
values = np.column_stack([np.linalg.eigvalsh(S@A@S) for A in
                          (env, total, total+cfg['Lambda0'])])
ranks = [int(np.sum(values[:, i] > cfg['rank_tol']*np.max(values[:, i]))) for i in range(3)]
print('Fig. 3b: environment / +BS / +prior ranks:', ranks)
print('Fig. 3a: known/unknown gain range CRB at 200 MHz (m):',
      *[np.interp(200e6, B, y) for y in (known, unknown)])

# Plotting: the displayed 1e-12 floor is not used for rank decisions.
colors = isac.plot_style(diagnostic=True)
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
ax[0].plot(B/1e6, known, color=colors[0], linewidth=1.8, label='Known complex gain')
ax[0].plot(B/1e6, unknown, color=colors[1], linewidth=1.8, linestyle='--', label='Unknown complex gain')
ax[0].set(xscale='log', yscale='log', xlim=(.1, 1000), ylim=(1e-5, 100),
          xlabel='Occupied bandwidth (MHz)', ylabel='Range bound (m)', title='(a)')
ax[0].legend(loc='lower left', fontsize=7)
for i, label in enumerate(['Sensing (rank 8)', '+ BS (rank 10)', '+ prior (rank 11)']):
    ax[1].plot(np.arange(1, len(values)+1), np.maximum(values[:, i], 1e-12),
               marker=['o', 's', '^'][i], markersize=2.5, linewidth=1.8,
               color=colors[i], linestyle=['-', '--', ':'][i], label=label)
ax[1].set(yscale='log', xlim=(.7, 11.3), ylim=(1e-13, 100), xticks=np.arange(1, 12, 2),
          xlabel='Ordered eigenvalue index', ylabel='Scaled information eigenvalue', title='(b)')
ax[1].text(.03, .17, 'Plot floor: $10^{-12}$', transform=ax[1].transAxes, fontsize=7)
ax[1].legend(loc='lower right', fontsize=7)
for a in ax:
    a.grid(True)
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig02_identifiability'))
