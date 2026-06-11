"""Real-time observability-aware control (RT-OAC).

A focused testbed for solving the observability-aware control (OAC) problem fast
enough for real-time, receding-horizon use. The verified mathematical core --- the
Short-Time Local Observability Gramian, Lie-derivative recursion, integrator, and
system models --- is *vendored verbatim* from the companion repo under ``src/`` (the
``observability_aware_control`` and ``example_lib`` packages), so they import normally
with no editable dependency or ``sys.path`` bootstrap. The companion's heavy
GUI/CUDA/symbolic dependencies are not pulled in: rt-oac declares only what the vendored
modules actually import (see ``pyproject.toml``).

``COMPANION_SRC`` still points at a sibling companion checkout if one exists (override
with ``OAC_SRC``); it is used only to locate optional *reference data* (e.g. the
companion's closed-loop ``.npz`` snapshots), not to put code on the path.

Importing this package also enables JAX 64-bit mode (required for the eigenvalue /
observability-Gramian computations to be numerically meaningful) and a persistent XLA
compilation cache.
"""

import os
from pathlib import Path

# --- Optional sibling companion checkout, for reference data only -------------------
# Layout: <root>/rt-oac/src/rt_oac/ and <root>/observability_aware_control/src. The core
# is vendored under src/, so this is NOT used to import code; callers use it only to
# locate optional reference-data files in the companion's data/ dir (see profile_solve).
COMPANION_SRC = Path(
    os.environ.get(
        "OAC_SRC",
        Path(__file__).resolve().parents[3] / "observability_aware_control" / "src",
    )
)

# --- 64-bit precision is mandatory for observability/eigenvalue work ---------------
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

# --- Persistent XLA compilation cache ----------------------------------------------
# The order-5 STLOG objective+gradient graph takes ~200 s to compile; without a
# persistent cache every fresh process re-pays it. Cache to <repo>/.cache (gitignored)
# so re-runs and the closed-loop loop start fast. Override dir with RT_OAC_CACHE_DIR.
_CACHE_DIR = Path(
    os.environ.get("RT_OAC_CACHE_DIR", Path(__file__).resolve().parents[2] / ".cache")
)
jax.config.update("jax_compilation_cache_dir", str(_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

CACHE_DIR = _CACHE_DIR
"""Directory holding the persistent XLA compilation cache."""

#: ``COMPANION_SRC`` (set above) optionally points at a sibling companion checkout's
#: source tree, used only to locate reference-data files (the core itself is vendored).
__all__ = ["CACHE_DIR", "COMPANION_SRC"]
