"""Structured run-record dump for the progress-report pipeline.

The headline examples render rerun ``.rrd`` recordings and matplotlib ``.png`` figures
for interactive inspection, but a *report* needs machine-readable data it can tabulate
and re-plot across configurations. When the Hydra key ``report`` is set to a directory,
each example calls :func:`dump` after its run, writing two sibling files keyed by a
config ``tag``:

* ``<tag>.npz`` -- every per-tick array the run recorded (trajectories, error, NEES,
  distance, observability, plan time);
* ``<tag>.json`` -- the scalar summary metrics plus config metadata (kind/variant/flags/
  horizon/band), the single source the tabulation code reads.

``report/figures.py`` consumes the ``.npz`` arrays; ``report/tabulate.py`` consumes the
``.json`` summaries. Keeping the schema here keeps the consumers in lock-step with the
producers (the examples).
"""

import json
import pathlib

import numpy as np


def dump(report_dir, tag, *, summary, arrays):
    """Write ``<report_dir>/<tag>.{npz,json}`` and return the ``.npz`` path.

    ``summary`` is a flat dict of JSON-serializable scalars/metadata; ``arrays`` is a
    dict of array-likes (each coerced with ``np.asarray``). The dir is made if absent.
    """
    out = pathlib.Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    meta = {"tag": tag, **summary}
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=2, default=float))
    return out / f"{tag}.npz"
