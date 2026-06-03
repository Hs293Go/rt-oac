"""Optimizers for the observability-aware control problem.

``scipy_trust_constr`` is the baseline path (reuses the fixed controller); a JAX-native
SQP/iLQR is a documented stretch goal pursued only if the scipy boundary dominates.
"""
