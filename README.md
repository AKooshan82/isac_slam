# ISAC sensing and allocation experiments

This repository contains matching Python and MATLAB implementations of eight self-contained experiments. Each script follows the same path: **set parameters → compute the model or run trials → draw a PDF plot and print a compact numerical summary**. Shared mathematics lives in one file per language, so an experiment can be read without searching through a framework.

## The idea behind the code

A mobile array observes landmarks and a fixed base station through a known pilot. Geometry maps position and heading to delays and bearings; array and waveform models turn those quantities into complex responses. The code then removes unknown channel gains when forming an equivalent Fisher information matrix (EFIM). That matrix measures what the observations say about the pose and map.

Some experiments use the full raw-pilot response and a multistart Levenberg–Marquardt MAP fit. Others work with information matrices to compare energy allocations, rate constraints, and the value of map knowledge. Randomized experiments use fixed seeds, paired trials where appropriate, bootstrap intervals for continuous metrics, and Wilson intervals for proportions.

## Layout

```text
python/
  isac_common.py          shared model, solvers, statistics, and plot style
  fig2_*.py … fig9_*.py   one script per experiment
matlab/
  isac.m                  equivalent static-method container
  fig2_*.m … fig9_*.m     one script per experiment
requirements.txt          Python dependencies
```

| Script | What it explores |
| --- | --- |
| `fig2_raw_estimation` | Raw-pilot MAP fitting, a feature surrogate, uncertainty, and convergence |
| `fig3_gain_and_gauge` | Information lost to unknown gains and recovered by a fixed base station |
| `fig4_waveform_moments` | How frequency and aperture spread affect delay and angle information |
| `fig5_full_vs_additive` | Full multipath information versus adding isolated-path approximations |
| `fig6_map_value` | The immediate and persistent value of map precision |
| `fig7_allocation` | Belief-aware energy allocation and rate-feasible alternatives |
| `fig8_rate_and_predictive` | Rate–localization tradeoffs and two-step planning |
| `fig9_discrete_and_robust` | Discrete selection and sensitivity to model uncertainty |

## Run an experiment

For Python, install the packages in a virtual environment and run a script from the `python/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cd python
python3 fig3_gain_and_gauge.py
```

The allocation scripts use **CVXPY with MOSEK**. MOSEK must be installed and licensed. For MATLAB, install **CVX with MOSEK**, add both to the MATLAB path, then run a script from `matlab/`:

```matlab
cd matlab
fig3_gain_and_gauge
```

Each script writes a PDF beside itself and prints the values used to inspect its result. The scripts do not need input datasets. To try a short run of the randomized experiments, set `QUICK = True` in Python or `QUICK = true` in MATLAB at the top of `fig2`, `fig5`, `fig7`, or `fig9`. This changes only the trial count; it does not switch models or solvers. Leave `QUICK` false for the full run. Generated PDFs and local environments are ignored by Git.

The code has been organized and checked statically, but the simulations and cross-language numerical comparisons have **not** been run as part of this cleanup.
