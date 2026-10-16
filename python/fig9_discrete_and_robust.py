"""Figure 9: generic greedy selection and paired SI/power robustness."""
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# QUICK changes only the two stochastic trial counts.
QUICK = False
DISCRETE_TRIALS = 20 if QUICK else 250
PHASE_TRIALS = 20 if QUICK else 2000
SI_DB = np.linspace(-30., 20., 101)
MISMATCH_DB = np.linspace(-15., 10., 101)
cfg = isac.defaults()


def logdet(A):
    """D-optimal information gain, Section V-C."""
    return np.linalg.slogdet(A)[1]


def gain(Lambda, Z, e):
    return logdet(isac.post(Lambda, Z, e))-logdet(Lambda)


# Computation, panel (a): generic PSD atoms, no communication constraint.
all_sets = {K: list(combinations(range(12), K)) for K in range(1, 6)}
ratios = np.empty((DISCRETE_TRIALS, 5))
for trial in range(1, DISCRETE_TRIALS+1):
    rng = np.random.Generator(np.random.MT19937(cfg['master_seed']+900000+trial))
    A = rng.standard_normal((6, 6))
    Lambda = A@A.T+2*np.eye(6)
    atoms = np.stack([(0.15+rng.random())*np.outer(v, v)/6
                      for v in (rng.standard_normal(6) for _ in range(12))], axis=2)
    base, current, selected = logdet(Lambda), Lambda.copy(), []
    for K in range(1, 6):
        remaining = [r for r in range(12) if r not in selected]
        chosen = max(remaining, key=lambda r: logdet(current+atoms[:, :, r])-base)
        selected.append(chosen)
        current += atoms[:, :, chosen]
        greedy = logdet(current)-base
        optimum = max(logdet(Lambda+atoms[:, :, list(ix)].sum(axis=2))-base for ix in all_sets[K])
        ratios[trial-1, K-1] = greedy/optimum
mean_ratio, low_ratio, high_ratio = isac.bootstrap_mean(ratios, cfg['master_seed']+950000)
minimum = ratios.min(axis=0)

# Computation, panels (b,c): design on full phase-averaged atoms.
w = isac.beams(cfg['q0'], cfg)
Z, _, meta = isac.efim_atoms(cfg['q0'], w, cfg)
g = meta['g']
Rmin = .5*isac.rate(isac.waterfill(g, cfg['Emax']), g, cfg)
nominal = isac.solve(cfg['Lambda0'], Z, g, Rmin, 'joint', cfg)
ordered = []
for db in [-12., -6., 0.]:
    c = isac.defaults()
    c['env_power'][2] *= 10**(db/10)
    ordered.append(isac.efim_atoms(c['q0'], w, c)[0])
robust = isac.solve_robust(cfg['Lambda0'], ordered, g, Rmin, cfg)
nonordered = []
for j in range(cfg['J']):
    c = isac.defaults()
    c['env_power'][j] *= 10**(-12/10)
    nonordered.append(isac.efim_atoms(c['q0'], w, c)[0])
robust_nonordered = isac.solve_robust(cfg['Lambda0'], nonordered, g, Rmin, cfg)
si_allocations = []
for db in SI_DB:
    c = isac.defaults()
    c['noiseS'] = 1+10**(db/10)
    Zsi = isac.efim_atoms(c['q0'], w, c)[0]
    si_allocations.append(isac.solve(cfg['Lambda0'], Zsi, g, Rmin, 'joint', cfg))

# Held-out conditional phases are shared by all policies and settings.
basis = isac.model_basis(cfg['q0'], w, cfg)
si_out = np.empty((PHASE_TRIALS, len(SI_DB), 2))
power_out = np.empty((PHASE_TRIALS, len(MISMATCH_DB), 3))
for trial in range(1, PHASE_TRIALS+1):
    rng = np.random.Generator(np.random.MT19937(cfg['master_seed']+1200000+trial))
    gains = dict(env=np.sqrt(cfg['env_power'])[:, None]*np.exp(2j*np.pi*rng.random((cfg['J'], cfg['R']))),
                 bs=np.sqrt(cfg['bs_power'])*np.exp(2j*np.pi*rng.random(cfg['R'])))
    _, Zs, Zb = isac.atoms_from_basis(basis, gains)
    for k, db in enumerate(SI_DB):
        actual = Zs/(1+10**(db/10))+Zb
        si_out[trial-1, k] = [gain(cfg['Lambda0'], actual, nominal),
                              gain(cfg['Lambda0'], actual, si_allocations[k])]
    for k, db in enumerate(MISMATCH_DB):
        changed = dict(env=gains['env'].copy(), bs=gains['bs'])
        changed['env'][2] *= 10**(db/20)
        actual, _, _ = isac.atoms_from_basis(basis, changed)
        power_out[trial-1, k] = [gain(cfg['Lambda0'], actual, e)
                                  for e in (nominal, robust, robust_nonordered)]
si_mean, si_low, si_high = isac.bootstrap_mean(si_out[:, :, 1]-si_out[:, :, 0],
                                                cfg['master_seed']+1299999)
power_mean, power_low, power_high = isac.bootstrap_mean(power_out[:, :, 1]-power_out[:, :, 0],
                                                        cfg['master_seed']+1299999)
print('Fig. 9a: greedy/exhaustive mean ratio by K:', mean_ratio)
print('Fig. 9a: minimum individual ratio:', ratios.min())
print('Fig. 9b: SI-aware minus nominal at ISR 20 dB (nat), 95% CI:',
      si_mean[-1], si_low[-1], si_high[-1])
for db in [-12., 0.]:
    k = np.argmin(abs(MISMATCH_DB-db))
    print(f'Fig. 9c: robust minus nominal at {db:g} dB (nat), 95% CI:',
          power_mean[k], power_low[k], power_high[k])
print('Fig. 9c: maximum ordered-versus-nonordered robust mean difference (nat):',
      np.max(abs(power_out[:, :, 1].mean(axis=0)-power_out[:, :, 2].mean(axis=0))))

# Plotting: original three panels, palette, bounds and scenario markers.
colors = isac.plot_style(diagnostic=True)
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6))
K = np.arange(1, 6)
ax[0].errorbar(K, mean_ratio, yerr=[mean_ratio-low_ratio, high_ratio-mean_ratio],
               color=colors[0], linewidth=1.8, marker='o', markersize=3, capsize=2,
               label='Mean + 95% CI')
ax[0].plot(K, minimum, '--s', color=colors[1], linewidth=1.8, markersize=3, label='Minimum')
ax[0].axhline(1-1/np.e, color=colors[3], linestyle=':', linewidth=1, label='1 - 1/e guarantee')
ax[0].set(xlim=(.8, 5.2), ylim=(.6, 1.04), xticks=K,
          xlabel='Selected actions K', ylabel='Greedy / optimum gain', title='(a)')
ax[0].legend(fontsize=6)
ax[1].fill_between(SI_DB, si_low, si_high, color='#CCE0EC', label='95% paired CI')
ax[1].plot(SI_DB, si_mean, color=colors[0], linewidth=1.8, label='Held-out phase mean')
ax[1].axhline(0, color=colors[3], linestyle=':', linewidth=1)
ax[1].set(xlim=(-30, 20), ylim=(-.02, .36), xlabel='Residual SI / noise (dB)',
          ylabel='SI-aware - nominal gain (nat)', title='(b)')
ax[1].legend(loc='upper left', fontsize=6)
ax[2].fill_between(MISMATCH_DB, power_low, power_high, color='#F3D9C6', label='95% paired CI')
ax[2].plot(MISMATCH_DB, power_mean, color=colors[1], linewidth=1.8, label='Both robust designs')
ax[2].axhline(0, color=colors[3], linestyle=':', linewidth=1)
for x in [-12, 0]:
    ax[2].axvline(x, color=colors[3], linestyle=':', linewidth=1)
ax[2].set(xlim=(-15, 10), ylim=(-.25, .13), xlabel='Landmark-3 power mismatch (dB)',
          ylabel='Robust - nominal gain (nat)', title='(c)')
ax[2].legend(loc='upper right', fontsize=6)
for a in ax:
    a.grid(True)
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig08_discrete_robustness'))
