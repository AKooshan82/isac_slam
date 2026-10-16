"""Figure 4: within-action frequency and angular moment laws."""
import numpy as np
import matplotlib.pyplot as plt

import isac_common as isac


# The 16-tone clusters remain on the 128-tone approved band grid.
cfg = isac.defaults()
cluster_offsets = np.arange(57)
theta = np.deg2rad(15.)
beam_offsets = np.linspace(0., 10., 301)

# Computation: centered frequency spread and fixed-energy angular EFIM.
spectral = np.empty(len(cluster_offsets))
for k, m in enumerate(cluster_offsets):
    bins = np.r_[np.arange(-8, 8)-m, np.arange(-8, 8)+m]
    f = bins*cfg['delta_f']
    spectral[k] = np.mean((f-f.mean())**2)/cfg['bandwidth']**2
at, dat = isac.ula(cfg['Nt'], theta)
ar, dar = isac.ula(cfg['Nr'], theta)
rx = np.vdot(dar-ar*np.vdot(ar, dar), dar-ar*np.vdot(ar, dar)).real
spatial = np.empty((len(beam_offsets), 4))
for k, offset in enumerate(beam_offsets):
    off = np.deg2rad(offset)
    w = np.column_stack([isac.ula(cfg['Nt'], theta-off)[0],
                         isac.ula(cfg['Nt'], theta+off)[0]])
    c, cp = at.conj()@w, dat.conj()@w
    power = abs(c)**2
    gain, weights, chi = power.mean(), power/power.sum(), cp/c
    tx = np.sum(weights*abs(chi-np.dot(weights, chi))**2).real
    spatial[k] = [rx+tx, 2*cfg['pilot_useful_fraction']*gain*(rx+tx),
                  2*cfg['pilot_useful_fraction']*gain*rx,
                  2*cfg['pilot_useful_fraction']*gain*tx]
print('Fig. 4b: normalized angular factor endpoints:', spatial[0, 0], spatial[-1, 0])
print('Fig. 4b: actual fixed-energy angular information endpoints:', spatial[0, 1], spatial[-1, 1])
print('Fig. 4a: maximum cluster offset / bandwidth:', cluster_offsets[-1]/cfg['N'])

# Plotting uses the original axes, labels, palette and line roles.
colors = isac.plot_style(diagnostic=True)
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
ax[0].plot(cluster_offsets/cfg['N'], spectral, color=colors[0], linewidth=1.8,
           label='Within-band compound action')
ax[0].axhline(.25, color=colors[3], linestyle=':', linewidth=1, label='Support upper bound')
ax[0].set(xlim=(0, .45), ylim=(0, .27), xlabel='Cluster-center offset / bandwidth',
          ylabel='Frequency variance / bandwidth$^2$', title='(a)')
ax[0].legend(loc='upper right', fontsize=7)
for j, label in enumerate(['Total (fixed energy)', 'Receive aperture', 'Transmit diversity'], start=1):
    ax[1].plot(beam_offsets, spatial[:, j], color=colors[j-1],
               linestyle=['-', '--', ':'][j-1], linewidth=1.8, label=label)
ax[1].set(xlim=(0, 10), ylim=(0, 15), xlabel='Within-action beam offset (deg)',
          ylabel='Angular information per energy', title='(b)')
ax[1].legend(loc='center left', fontsize=7)
for a in ax:
    a.grid(True)
fig.tight_layout()
print('Saved', isac.save_pdf(fig, __file__, 'fig03_waveform_moments'))
