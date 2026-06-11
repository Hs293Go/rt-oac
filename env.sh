# Optional convenience for running RT-OAC scripts:
#
#   source env.sh && uv run python benchmarks/profile_solve.py
#
# The companion core is vendored under src/, so imports work without any PYTHONPATH setup
# -- you can just `uv run python ...` directly. The only thing this adds is JAX_PLATFORMS=cpu,
# which avoids the incomplete-CUDA noise in the shared environment and matches the
# Jetson/CPU real-time target. (OAC_SRC is honored by rt_oac only to locate optional
# reference-data files in a sibling companion checkout, never to import code.)
export JAX_PLATFORMS=cpu
