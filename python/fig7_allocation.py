"""Figure 7: belief-dependent energy and rate-feasible policy tradeoffs."""
import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# QUICK changes only the number of held-out phase trials.
QUICK = False
TRIALS = 20 if QUICK else 2000
L3_STD = np.logspace(np.log10(.2), np.log10(12.), 81)
cfg = isac.defaults()
w = isac.beams(cfg['q0'], cfg)
Z, _, meta = isac.efim_atoms(cfg['q0'], w, cfg)
g = meta['g']
Rmax = isac.rate(isac.waterfill(g, cfg['Emax']), g, cfg)
Rmin = .5*Rmax

# Computation: use full phase-averaged atoms for every design.
fractions = np.empty((len(L3_STD), cfg['R']))
for k, sigma in enumerate(L3_STD):
    Lambda = cfg['Lambda0'].copy()
    Lambda[7:9, 7:9] = np.eye(2)/sigma**2
    fractions[k] = isac.solve(Lambda, Z, g, Rmin, 'joint', cfg)/cfg['Emax']

policies = ['Joint D', 'Position only', 'Target conditional blocks',
            'Rate only', 'Uniform/rate mixture', 'Strongest reflector']
energy = np.empty((cfg['R'], 6))
for p, mode in enumerate(['joint', 'loc', 'target', 'rate']):
    energy[:, p] = isac.solve(cfg['Lambda0'], Z, g, Rmin, mode, cfg)
energy[:, 4], mixture = isac.feasible_uniform(g, Rmin, cfg)
lo, hi = 0., cfg['Emax']
for _ in range(60):
    mid = (lo+hi)/2
    if isac.rate(np.array([mid, cfg['Emax']-mid, 0., 0., 0.]), g, cfg) >= Rmin:
        hi = mid
    else:
        lo = mid
energy[:, 5] = [hi, cfg['Emax']-hi, 0., 0., 0.]
nominal = np.empty((6, 2))
for p in range(6):
    peb, _, meb = isac.metrics(isac.post(cfg['Lambda0'], Z, energy[:, p]))
    nominal[p] = [peb, meb.mean()]

# Held-out phases yield paired covariance comparisons; not estimator RMSE.
outcomes = np.empty((TRIALS, 6, 2))
for trial in range(1, TRIALS+1):
    rng = np.random.Generator(np.random.MT19937(cfg['master_seed']+600000+trial))
    gains = dict(env=np.sqrt(cfg['env_power'])[:, None]*np.exp(2j*np.pi*rng.random((cfg['J'], cfg['R']))),
                 bs=np.sqrt(cfg['bs_power'])*np.exp(2j*np.pi*rng.random(cfg['R'])))
    Zt, _, _ = isac.efim_atoms(cfg['q0'], w, cfg, gains)
    for p in range(6):
        peb, _, meb = isac.metrics(isac.post(cfg['Lambda0'], Zt, energy[:, p]))
        outcomes[trial-1, p] = [peb, meb.mean()]
paired = outcomes[:, 1]-outcomes[:, 0]
pair_mean, pair_lo, pair_hi = isac.bootstrap_mean(paired, cfg['master_seed']+900002)
print('Fig. 7a: L3 directed fraction endpoints:', fractions[0, 3], fractions[-1, 3])
print('Fig. 7b: joint / position-only / feasible-uniform PEB, MEB:',
      nominal[0], nominal[1], nominal[4])
print('Fig. 7b: uniform rate-only mixture fraction:', mixture)
print('Fig. 7b: held-out position-only minus joint PEB/MEB and paired 95% CI:',
      pair_mean, pair_lo, pair_hi)

# Plotting: original five energy curves and six PEB/MEB operating points.
colors = isac.plot_style()
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))
for r, name in enumerate(['BS', 'Landmark 1', 'Landmark 2', 'Landmark 3', 'Landmark 4']):
    ax[0].plot(L3_STD, fractions[:, r], color=colors[r], linewidth=1.4, label=name)
ax[0].set(xscale='log', xlim=(.2, 12), ylim=(0, .65),
          xlabel='Landmark-3 prior coordinate SD (m)', ylabel='Energy fraction', title='(a)')
ax[0].legend(loc='upper left', fontsize=7, ncol=2)
for p, name in enumerate(policies):
    ax[1].plot(nominal[p, 0], nominal[p, 1], marker='o', markersize=4,
               color=colors[p], linestyle='none', label=name)
ax[1].set(xlim=(.8, 1.32), ylim=(2.70, 3.75),
          xlabel='Position covariance metric PEB (m)',
          ylabel='Mean landmark covariance metric (m)', title='(b)')
ax[1].legend(loc='upper left', fontsize=6)
for a in ax:
    a.grid(True)
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig06_belief_baselines'))
