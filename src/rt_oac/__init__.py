"""Real-time observability-aware control (RT-OAC).

A focused testbed for solving the observability-aware control (OAC) problem fast
enough for real-time, receding-horizon use. The verified mathematical core --- the
Short-Time Local Observability Gramian, Lie-derivative recursion, integrator, and
system models --- is reused *verbatim* from the companion repo
``observability_aware_control``, declared as a local editable dependency (see
``[tool.uv.sources]`` in ``pyproject.toml``). The editable install exposes both of its
packages (``observability_aware_control`` and ``example_lib``), so they import
normally; its heavy GUI/CUDA/symbolic dependencies are excluded via
``[tool.uv] override-dependencies``.

As a fallback for running against an *uninstalled* checkout, the companion ``src``
directory (``OAC_SRC``, or the sibling ``observability_aware_control/src``) is appended
to ``sys.path`` if present.

Importing this package also enables JAX 64-bit mode (required for the eigenvalue /
observability-Gramian computations to be numerically meaningful) and a persistent XLA
compilation cache.
"""

import os
from pathlib import Path
import sys

# --- Fallback path for an uninstalled companion checkout ----------------------------
# Layout: <root>/rt-oac/src/rt_oac/ and <root>/observability_aware_control/src. With the
# editable dependency this is redundant (and skipped); it only matters if the package is
# not installed. Appended (not inserted) so the installed package wins.
COMPANION_SRC = Path(
    os.environ.get(
        "OAC_SRC",
        Path(__file__).resolve().parents[3] / "observability_aware_control" / "src",
    )
)
if COMPANION_SRC.is_dir() and str(COMPANION_SRC) not in sys.path:
    sys.path.append(str(COMPANION_SRC))

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

#: ``COMPANION_SRC`` (set above) points at the companion repo's source tree.
__all__ = ["CACHE_DIR", "COMPANION_SRC"]
