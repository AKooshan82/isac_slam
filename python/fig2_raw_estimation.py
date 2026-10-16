"""Figure 2: single-epoch raw-pilot MAP and idealized feature surrogate."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import isac_common as isac


# Experiment parameters (Section VII-A). QUICK changes the trial count only.
QUICK = False
TRIALS = 20 if QUICK else 2000
SNR_DB = np.arange(-10, 31, 5)

cfg = isac.defaults()
beam = isac.beams(cfg['q0'], cfg)  # Keep the five nominal beams.
cfg['q0'] = np.r_[35., 20., np.deg2rad(20.), 45., 55.]
cfg['env_power'], cfg['J'] = np.array([.085]), 1
cfg['P0'] = np.diag(np.r_[1., 1., np.deg2rad(6.), 1.5, 1.5]**2)
mu, P, e = cfg['q0'], cfg['P0'], np.full(5, cfg['Emax']/5)
_, _, b, _, _ = isac.raw_response(mu, beam, cfg)
reference = sum(e[r]*cfg['bs_power']*np.vdot(b[r], b[r]).real for r in range(5))

# Computation: one saved outcome is one independent truth/pilot/feature draw.
rmse, low, high, local = [], [], [], []
nonconvergence, coverage, solve_ms = [], [], []
for setting, snr in enumerate(SNR_DB, start=1):
    cfg['noiseS'] = cfg['noiseB'] = reference/10**(snr/10)
    _, _, precision = isac.feature_model(mu, beam, e, cfg)
    squared = np.empty((TRIALS, 3, 2))
    local_cov = np.empty((TRIALS, 3))
    failed = np.empty((TRIALS, 2), bool)
    covered = np.empty((TRIALS, 2), bool)
    elapsed = np.empty((TRIALS, 2))
    for trial in range(1, TRIALS+1):
        seed = cfg['master_seed']+20000000+setting*100000+trial*3
        rng = np.random.Generator(np.random.MT19937(seed))
        truth = mu+np.linalg.cholesky(P)@rng.standard_normal(5)
        obs, gains, meta = isac.observe(truth, e, beam, cfg, seed+1)
        raw = isac.raw_fit(mu, P, obs, beam, cfg)
        rng_feature = np.random.Generator(np.random.MT19937(seed+2))
        h, _, _ = isac.feature_model(truth, beam, e, cfg)
        z = h+rng_feature.standard_normal(len(h))/np.sqrt(precision)
        feature = isac.feature_fit(mu, P, z, precision, beam, e, cfg)
        for p, fit in enumerate((raw, feature)):
            values = isac.trial_metrics(fit['q'], truth, fit['P'])
            squared[trial-1, :, p] = values[:3]
            failed[trial-1, p] = fit['failed']
            covered[trial-1, p] = values[3] <= chi2.ppf(.95, 5)
            elapsed[trial-1, p] = fit['runtime']*1000
        Z, _, _ = isac.efim_atoms(truth, beam, cfg, gains)
        C = np.linalg.inv(isac.post(np.linalg.inv(P), Z, e))
        local_cov[trial-1] = [np.trace(C[:2, :2]), C[2, 2], np.trace(C[3:, 3:])]
        if meta['cp_violation']:
            raise ValueError('Trial violates the fixed CP assumption')
    centre, lo, hi = isac.bootstrap_rmse(squared, cfg['master_seed']+900+setting)
    rmse.append(centre)
    low.append(lo)
    high.append(hi)
    local.append(np.sqrt(local_cov.mean(axis=0)))
    nonconvergence.append([isac.wilson(failed[:, p].sum(), TRIALS) for p in range(2)])
    coverage.append([isac.wilson(covered[:, p].sum(), TRIALS) for p in range(2)])
    solve_ms.append(elapsed.mean(axis=0))

rmse, low, high, local = map(np.asarray, (rmse, low, high, local))
print('Fig. 2, 30 dB: raw/feature/local position RMSE (m):',
      rmse[-1, 0, 0], rmse[-1, 0, 1], local[-1, 0])
print('Fig. 2, 30 dB: raw/feature/local map RMSE (m):',
      rmse[-1, 2, 0], rmse[-1, 2, 1], local[-1, 2])
print('Fig. 2, 30 dB: raw/feature/local heading RMSE (deg):',
      np.rad2deg(rmse[-1, 1, 0]), np.rad2deg(rmse[-1, 1, 1]), np.rad2deg(local[-1, 1]))
print('Fig. 2: raw nonconvergence range:', min(x[0][0] for x in nonconvergence), max(x[0][0] for x in nonconvergence))
print('Fig. 2, 30 dB: raw/feature 95% joint coverage:', coverage[-1])
print('Fig. 2: raw/feature mean solve time ranges (ms):', np.min(solve_ms, axis=0), np.max(solve_ms, axis=0))

# Plotting: three panels, same labels, palette, shading, and log axis.
colors = isac.plot_style()
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
for panel, (ax, metric, ylabel, scale) in enumerate(zip(
        axes, [0, 2, 1], ['Position RMSE (m)', 'Landmark RMSE (m)', 'Heading RMSE (deg)'],
        [1., 1., 180/np.pi])):
    for policy, label in enumerate(['Raw MAP', 'Feature surrogate']):
        ax.fill_between(SNR_DB, low[:, metric, policy]*scale, high[:, metric, policy]*scale,
                        color=colors[policy], alpha=.18, linewidth=0)
        ax.plot(SNR_DB, rmse[:, metric, policy]*scale, color=colors[policy],
                linewidth=1.4, marker=['o', 's'][policy], markersize=2.5, label=label)
    ax.plot(SNR_DB, local[:, metric]*scale, color=colors[2], linestyle='--',
            linewidth=1.4, label='Raw local covariance')
    ax.set(title=f'({chr(97+panel)})', xlabel='BS integrated pilot SNR (dB)',
           ylabel=ylabel, yscale='log')
    ax.grid(True)
    ax.legend(fontsize=7, loc='lower left' if panel == 2 else 'upper right')
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig01_estimator_bounds'))
