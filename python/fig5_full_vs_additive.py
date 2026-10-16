"""Figure 5: full multipath atoms versus isolated-landmark addition."""
import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# Four frozen geometries; QUICK changes trial count, not the model or solver.
QUICK = False
TRIALS = 20 if QUICK else 2000
GAPS = [None, .05, .5, 5.]
LABELS = ['Nominal', '0.05 m', '0.5 m', '5 m']
master = isac.defaults()


def logdet_gain(Lambda, Z, e):
    """Joint D-optimal gain relative to the same prior."""
    return np.linalg.slogdet(isac.post(Lambda, Z, e))[1]-np.linalg.slogdet(Lambda)[1]


def terminal_peb(Lambda, Z0, e0, Z1, e1):
    """Two-epoch frozen-model PEB, evaluated with full atoms."""
    return isac.metrics(isac.post(isac.predict(isac.post(Lambda, Z0, e0), 2.), Z1, e1))[0]


# Computation. Each scenario rebuilds both epoch codebooks and its rate target.
outcomes = np.empty((4, TRIALS, 3))  # atom error, joint loss, PEB % penalty
absolute_penalty = np.empty((4, TRIALS))
for case, gap in enumerate(GAPS, start=1):
    cfg = isac.defaults()
    if gap is not None:
        p, m2 = cfg['q0'][:2], cfg['q0'][5:7]
        cfg['q0'][7:9] = m2+gap*(m2-p)/np.linalg.norm(m2-p)
        cfg['q2'][7:9] = cfg['q0'][7:9]
    w0, w1 = isac.beams(cfg['q0'], cfg), isac.beams(cfg['q2'], cfg)
    _, _, meta = isac.efim_atoms(cfg['q0'], w0, cfg)
    _, _, next_meta = isac.efim_atoms(cfg['q2'], w1, cfg)
    Rmin = .5*isac.rate(isac.waterfill(meta['g'], cfg['Emax']), meta['g'], cfg)
    for trial in range(1, TRIALS+1):
        seed = cfg['master_seed']+400000+case*10000+trial
        rng = np.random.Generator(np.random.MT19937(seed))
        gains0 = dict(env=np.sqrt(cfg['env_power'])[:, None]*np.exp(2j*np.pi*rng.random((cfg['J'], cfg['R']))),
                      bs=np.sqrt(cfg['bs_power'])*np.exp(2j*np.pi*rng.random(cfg['R'])))
        gains1 = dict(env=np.sqrt(cfg['env_power'])[:, None]*np.exp(2j*np.pi*rng.random((cfg['J'], cfg['R']))),
                      bs=np.sqrt(cfg['bs_power'])*np.exp(2j*np.pi*rng.random(cfg['R'])))
        Z0, Za0, _ = isac.efim_atoms(cfg['q0'], w0, cfg, gains0)
        Z1, Za1, _ = isac.efim_atoms(cfg['q2'], w1, cfg, gains1)
        S = np.diag(cfg['state_scale'])
        full = np.hstack([S@Z0[:, :, r]@S for r in range(cfg['R'])])
        add = np.hstack([S@Za0[:, :, r]@S for r in range(cfg['R'])])
        atom_error = np.linalg.norm(add-full, 'fro')/np.linalg.norm(full, 'fro')
        ejf = isac.solve(cfg['Lambda0'], Z0, meta['g'], Rmin, 'joint', cfg)
        eja = isac.solve(cfg['Lambda0'], Za0, meta['g'], Rmin, 'joint', cfg)
        joint_loss = logdet_gain(cfg['Lambda0'], Z0, ejf)-logdet_gain(cfg['Lambda0'], Z0, eja)
        epf0, epf1 = isac.solve_predictive(cfg['Lambda0'], Z0, meta['g'], Z1,
                                           next_meta['g'],
                                           2., Rmin, 'loc', cfg)
        epa0, epa1 = isac.solve_predictive(cfg['Lambda0'], Za0, meta['g'], Za1,
                                           next_meta['g'],
                                           2., Rmin, 'loc', cfg)
        pf = terminal_peb(cfg['Lambda0'], Z0, epf0, Z1, epf1)
        pa = terminal_peb(cfg['Lambda0'], Z0, epa0, Z1, epa1)
        absolute_penalty[case-1, trial-1] = pa-pf
        outcomes[case-1, trial-1] = [atom_error, joint_loss, 100*(pa-pf)/pf]

means, lows, highs = [], [], []
bootstrap_rng = np.random.default_rng(master['master_seed']+1300)
for case in range(4):
    values = outcomes[case]
    draws = np.array([values[bootstrap_rng.integers(TRIALS, size=TRIALS)].mean(axis=0)
                      for _ in range(master['bootstrap'])])
    mean = values.mean(axis=0)
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    means.append(mean)
    lows.append(lo)
    highs.append(hi)
    print(f'Fig. 5 {LABELS[case]}: mean atom error, conditional joint loss (nat), '
          f'terminal PEB penalty (m): {mean[0]:.8g}, {mean[1]:.8g}, '
          f'{absolute_penalty[case].mean():.8g}')
means, lows, highs = map(np.asarray, (means, lows, highs))

# Plotting: two original panels and paired whole-trial 95% error bars.
colors = isac.plot_style()
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))
x = np.arange(1, 5)
for a, col, color, ylabel, label in [
        (ax[0], 0, colors[0], 'Relative full/additive atom error', 'Mean and 95% CI'),
        (ax[1], 2, colors[1], 'Terminal PEB penalty (%)', 'Additive design, full evaluation')]:
    a.errorbar(x, means[:, col], yerr=[means[:, col]-lows[:, col], highs[:, col]-means[:, col]],
               color=color, linewidth=1.4, marker='o', markersize=3, capsize=2, label=label)
    a.set(xlim=(.6, 4.4), xticks=x, xticklabels=LABELS,
          xlabel='Nominal map / landmark-2--3 separation', ylabel=ylabel,
          title='(a)' if a is ax[0] else '(b)')
    a.grid(True)
    a.legend(fontsize=7)
ax[0].set_yscale('log')
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig04_separability'))
