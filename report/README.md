# Progress report — *Improving the Beginner's Observability-Aware Control*

A self-contained, reproducible progress report on two unambiguous advances to the JGCD baseline
OAC — real-time solve and a soft-constraint reformulation — with **carried estimation in the loop**
isolated as the open problem ([`../docs/carried_estimation_plan.md`](../docs/carried_estimation_plan.md)).
Source: [`improving_beginners_oac.tex`](improving_beginners_oac.tex).

## Build it

```bash
# from the repo root
bash report/sweep.sh                                   # 1. generate data/  (Hydra sweep + trichotomy)
JAX_PLATFORMS=cpu uv run python report/tabulate.py     # 2. data/*.json -> tables/*.tex
JAX_PLATFORMS=cpu uv run python report/figures.py      # 3. data/*.npz  -> figures/*.pdf
cd report && latexmk -pdf improving_beginners_oac.tex  # 4. -> improving_beginners_oac.pdf
```

## Pipeline

The report is data-driven end to end — no number is hand-entered.

| Stage | Script | Produces | Consumes |
|-------|--------|----------|----------|
| Sweep | [`sweep.sh`](sweep.sh) | `data/<tag>.{npz,json}` | the two headline examples via Hydra `report=` |
| Trichotomy | [`trichotomy.py`](trichotomy.py) | `data/quad_estimation_*.{npz,json}`, `data/trichotomy.json` | the quad example across 3 estimator setups × seeds |
| Tabulate | [`tabulate.py`](tabulate.py) | `tables/{solve,planar,trichotomy}.tex` | `data/*.json` |
| Figures | [`figures.py`](figures.py) | `figures/{quad_orbit,planar_spatial,planar_timeseries,quad_trichotomy}.pdf` | `data/*.npz` |
| Typeset | `improving_beginners_oac.tex` | the PDF | `tables/*.tex`, `figures/*.pdf` |

The data hook is [`rt_oac.report.dump`](../src/rt_oac/report.py): with the Hydra key `report=<dir>`,
each run drops a `<tag>.npz` (per-tick arrays) and a `<tag>.json` (scalar summary + config
metadata), keeping producers and consumers in lock-step.

## Configurations

| Tag | Model | What it shows |
|-----|-------|---------------|
| `quad_open` / `_soft` | quadrotor | real-time open-loop orbit; hard vs soft constraint |
| `planar_{logdet,hybrid}` / `_soft` | planar | OAC vs no-OAC (carried EKF, robustly observable); soft constraint |
| `quad_estimation{,_reanchor,_planontruth}` | quadrotor | the carried-estimation **trichotomy** (re-anchor / carry-plan-on-truth / closed loop) — future work |

The trichotomy is the diagnostic that isolates the #8 divergence: re-anchoring the EKF mean to
truth (the paper's validation) is robustly consistent; *carrying* the mean and *closing* the loop
on it are what destabilize. See the report's future-work section and the plan doc.

Generated artifacts (`data/`, the report PDF, LaTeX aux) are git-ignored; the `.tex`, the four
scripts, and `sweep.sh` are the reproducible source.
