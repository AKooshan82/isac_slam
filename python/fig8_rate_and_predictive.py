"""Figure 8: ideal-rate frontier and fixed-model two-epoch covariance design."""
import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# Deterministic grids and the separately reported weak-BS sensitivity.
cfg = isac.defaults()
RATE_FRACTIONS = np.linspace(0., .95, 41)
POSITION_PROCESS_SD = np.linspace(.25, 5., 31)
BS_POWERS = [1.2, .08]

# Computation, panel (a): every point optimizes joint D-optimality.
w0 = isac.beams(cfg['q0'], cfg)
Z0, _, meta0 = isac.efim_atoms(cfg['q0'], w0, cfg)
g0 = meta0['g']
Rmax = isac.rate(isac.waterfill(g0, cfg['Emax']), g0, cfg)
Rmin = .5*Rmax
frontier = np.empty((len(RATE_FRACTIONS), 3))
for k, fraction in enumerate(RATE_FRACTIONS):
    e = isac.solve(cfg['Lambda0'], Z0, g0, fraction*Rmax, 'joint', cfg)
    peb, _, meb = isac.metrics(isac.post(cfg['Lambda0'], Z0, e))
    frontier[k] = [fraction*Rmax, peb, meb.mean()]

# Computation, panel (b): matched first-epoch and terminal objectives.
rows = []
for bs_power in BS_POWERS:
    c = isac.defaults()
    c['bs_power'] = bs_power
    A0, _, m0 = isac.efim_atoms(c['q0'], isac.beams(c['q0'], c), c)
    A1, _, m1 = isac.efim_atoms(c['q2'], isac.beams(c['q2'], c), c)
    target = .5*isac.rate(isac.waterfill(m0['g'], c['Emax']), m0['g'], c)
    myopic_loc = isac.solve(c['Lambda0'], A0, m0['g'], target, 'loc', c)
    myopic_joint = isac.solve(c['Lambda0'], A0, m0['g'], target, 'joint', c)
    uniform0, _ = isac.feasible_uniform(m0['g'], target, c)
    uniform1, _ = isac.feasible_uniform(m1['g'], target, c)
    for qstd in POSITION_PROCESS_SD:
        energies = []
        for e0, mode in [(myopic_loc, 'loc'), (myopic_joint, 'joint')]:
            Lpred = isac.predict(isac.post(c['Lambda0'], A0, e0), qstd)
            e1 = isac.solve(Lpred, A1, m1['g'], target, mode, c)
            energies.append((e0, e1))
        for mode in ['loc', 'joint']:
            energies.append(isac.solve_predictive(c['Lambda0'], A0, m0['g'], A1,
                                                  m1['g'], qstd, target, mode, c))
        energies.append(isac.solve_predictive(c['Lambda0'][:3, :3], A0[:3, :3], m0['g'],
                                              A1[:3, :3], m1['g'], qstd, target, 'loc', c))
        energies.append((uniform0, uniform1))
        for policy, (e0, e1) in enumerate(energies, start=1):
            if policy == 5:
                L, Zfirst, Zsecond = c['Lambda0'][:3, :3], A0[:3, :3], A1[:3, :3]
            else:
                L, Zfirst, Zsecond = c['Lambda0'], A0, A1
            Lplus = isac.post(L, Zfirst, e0)
            terminal = isac.post(isac.predict(Lplus, qstd), Zsecond, e1)
            peb, _, meb = isac.metrics(terminal)
            ablation = np.nan
            if policy != 5:
                Lab = isac.ablate_map(Lplus, L)
                ablation = isac.metrics(isac.post(isac.predict(Lab, qstd), Zsecond, e1))[0]-peb
            rows.append((bs_power, qstd, policy, peb, meb.mean(), e0[3]/c['Emax'], ablation))
rows = np.asarray(rows)
nominal = rows[rows[:, 0] == 1.2]
weak = rows[rows[:, 0] == .08]
def range_gain(data):
    base = data[data[:, 2] == 1, 3]
    prediction = data[data[:, 2] == 3, 3]
    return np.min(100*(base-prediction)/base), np.max(100*(base-prediction)/base)

print('Fig. 8a: rate-only max and half-max target (bit/s/Hz):', Rmax, Rmin)
print('Fig. 8a: PEB at zero/half/95% rate target:', frontier[0, 1],
      frontier[np.argmin(abs(RATE_FRACTIONS-.5)), 1], frontier[-1, 1])
print('Fig. 8a: mean MEB endpoints:', frontier[0, 2], frontier[-1, 2])
print('Fig. 8b: predictive-versus-myopic-localization PEB reduction % nominal/weak:',
      range_gain(nominal), range_gain(weak))
print('Fig. 8b: first-epoch L3-directed fraction range:',
      np.min(rows[rows[:, 2] == 3, 5]), np.max(rows[rows[:, 2] == 3, 5]))
print('Fig. 8b: conditional-map-precision ablation PEB increase range nominal/weak:',
      (np.nanmin(nominal[:, 6]), np.nanmax(nominal[:, 6])),
      (np.nanmin(weak[:, 6]), np.nanmax(weak[:, 6])))

# Plotting follows the original axes, labels and policy line roles.
colors = isac.plot_style()
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))
ax[0].plot(frontier[:, 0]*1000, frontier[:, 1], color=colors[0], linewidth=1.4, label='PEB')
ax[0].plot(frontier[:, 0]*1000, frontier[:, 2], color=colors[1], linewidth=1.4,
           linestyle='--', label='Mean MEB')
ax[0].set(ylim=(.7, 3.35), xlabel='Ideal-CSI rate target ($10^{-3}$ bit/s/Hz)',
          ylabel='Covariance metric (m)', title='(a)')
ax[0].legend(loc='center left', fontsize=7)
for policy, label in enumerate(['Myopic position', 'Myopic joint D', 'Terminal position P2',
                                'Terminal joint D', 'Known-map oracle'], start=1):
    part = nominal[nominal[:, 2] == policy]
    ax[1].plot(part[:, 1], part[:, 3], color=colors[policy-1],
               linestyle=['-', '--', ':'][((policy-1) % 3)], linewidth=1.4, label=label)
ax[1].set(xlabel='Position process SD (m)', ylabel='Terminal PEB (m)',
          title='(b) nominal BS power 1.2')
ax[1].legend(loc='lower right', fontsize=6)
for a in ax:
    a.grid(True)
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig07_comm_predictive'))
