r"""Build the report's ablation tables (booktabs LaTeX) from ``report/data/*.json``.

Each headline-example run drops a ``<tag>.json`` summary (see ``rt_oac.report.dump``); this
script indexes them by tag and emits three ``tabular`` fragments, one per open problem cluster,
into ``report/tables/``:

* ``solve.tex``      -- per-tick solve time + observability (real-time solve and the
  hard-vs-soft constraint), quadrotor open-loop orbit;
* ``planar.tex``     -- planar leader-follower OAC vs no-OAC with a carried EKF, across the
  log-det / balanced and hard / soft variants;
* ``trichotomy.tex`` -- the carried-estimation future-work diagnostic (re-anchor / carry /
  closed loop), from ``report/data/trichotomy.json``.

The fragments are ``\\input`` by the report's table floats. Missing metrics render as ``---`` so
a partial sweep still produces well-formed tables. Run from the repo root:

    JAX_PLATFORMS=cpu uv run python report/tabulate.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
TABLES = HERE / "tables"

DASH = "---"


def load():
    """Index every ``report/data/*.json`` summary by its ``tag``."""
    out = {}
    for path in sorted(DATA.glob("*.json")):
        d = json.loads(path.read_text())
        if "tag" in d:  # skip non-run summaries (e.g. trichotomy.json)
            out[d["tag"]] = d
    return out


def num(summ, tag, key, spec="{:.2f}", scale=1.0):
    """Formatted metric ``summ[tag][key]`` (scaled), or ``---`` if absent."""
    val = summ.get(tag, {}).get(key)
    if val is None:
        return DASH
    return spec.format(val * scale)


def band(summ, tag):
    """The realized distance range ``[min, max]`` as a LaTeX cell, or ``---``."""
    s = summ.get(tag)
    if not s or s.get("dist_min") is None:
        return DASH
    return f"$[{s['dist_min']:.2f},\\,{s['dist_max']:.2f}]$"


def write(name, body):
    out = TABLES / name
    out.write_text(body)
    print(f"wrote {out}")


def solve_table(s):
    """Problem 1 + 4: real-time solve and the soft-vs-hard constraint (quadrotor orbit)."""
    rows = [
        # Cited baseline: the paper runs the identical OPC offline (Table 1 therein).
        r"JGCD baseline OPC\textsuperscript{a} & min-eig & trust-constr "
        r"& hard & $2000$--$4000$ & --- & --- & $6/6$ & $[1,3]$ \\",
    ]
    spec = [
        ("Open-loop orbit", "quad_open", "hard"),
        ("Open-loop orbit", "quad_open_soft", "soft"),
    ]
    for label, tag, mode in spec:
        rows.append(
            f"{label} & soft-min & SLSQP@6 & {mode} "
            f"& {num(s, tag, 'plan_median_ms', '{:.0f}')} "
            f"& {num(s, tag, 'plan_p95_ms', '{:.0f}')} "
            f"& {num(s, tag, 'plan_tick0_ms', '{:.1f}', 1e-3)} "
            f"& ${num(s, tag, 'ndir_median', '{:.0f}')}/6$ "
            f"& {band(s, tag)} \\\\"
        )
    body = "\n".join([
        r"\begin{tabular}{llllrrrcc}",
        r"\toprule",
        r"Case & Objective & Solver & Constr. & Median & p95 & Compile "
        r"& Obs.\ dirs & $d$ range \\",
        r" & & & & [ms] & [ms] & [s] & & [m] \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    write("solve.tex", body)


def trichotomy_table():
    """Future work: the carried-estimation trichotomy from report/data/trichotomy.json."""
    path = DATA / "trichotomy.json"
    if not path.exists():
        write(
            "trichotomy.tex", "% trichotomy.json missing -- run report/trichotomy.py\n"
        )
        return
    t = json.loads(path.read_text())
    order = [
        (
            "reanchor",
            "Re-anchor mean to truth (paper)",
            "truth (pinned)",
            "$\\approx$truth",
        ),
        ("planontruth", "Carry mean, plan on truth", "carried", "truth"),
        ("closed", "Carry mean, plan on estimate", "carried", "estimate"),
    ]
    rows = []
    for key, label, mean, ctrl in order:
        d = t["setups"].get(key, {})
        diverged = d.get("diverged_any")
        flag = DASH if diverged is None else (r"\checkmark" if diverged else r"--")
        rows.append(
            f"{label} & {mean} & {ctrl} "
            f"& {d.get('nees_median', float('nan')):.0f} "
            f"& {d.get('nees_worst', float('nan')):.0f} "
            f"& {d.get('errmax_median', float('nan')):.2f} "
            f"& {flag} \\\\"
        )
    band = t.get("nees_band", [2.7, 19.0])
    body = "\n".join([
        r"\begin{tabular}{lllrrrc}",
        r"\toprule",
        r"Setup & EKF mean & Plans on & NEES & NEES & $\|e\|_{\max}$ & Div. \\",
        r" & & & med. & worst & med.\ [m] & \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        f"% chi^2_9 band [{band[0]:.1f}, {band[1]:.1f}]; NEES<band = conservative, >>band = divergent",
    ])
    write("trichotomy.tex", body)


def planar_table(s):
    """Problem 2 + 3 + 4: planar OAC vs no-OAC, log-det/balanced x hard/soft."""
    ref = s.get("planar_logdet", {})  # no-OAC baseline is scheme-independent
    rows = [
        f"No-OAC (drive straight) & {num(s, 'planar_logdet', 'noac_rmse_m')} "
        f"& {num(s, 'planar_logdet', 'noac_final_m')} & --- & --- \\\\",
        r"\midrule",
    ]
    spec = [
        ("OAC log-det", "planar_logdet"),
        ("OAC log-det, soft", "planar_logdet_soft"),
        ("OAC balanced", "planar_hybrid"),
        ("OAC balanced, soft", "planar_hybrid_soft"),
    ]
    for label, tag in spec:
        rows.append(
            f"{label} & {num(s, tag, 'oac_rmse_m')} "
            f"& {num(s, tag, 'oac_final_m')} "
            f"& {num(s, tag, 'plan_median_ms', '{:.1f}')} "
            f"& {num(s, tag, 'mineig_median', '{:.2f}')} \\\\"
        )
    _ = ref
    body = "\n".join([
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Configuration & RMSE & Final $\|e\|$ & Median solve "
        r"& $\lambda_{\min}$ \\",
        r" & [m] & [m] & [ms] & \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    write("planar.tex", body)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    s = load()
    print(f"loaded {len(s)} summaries: {', '.join(sorted(s))}")
    solve_table(s)
    planar_table(s)
    trichotomy_table()


if __name__ == "__main__":
    main()
