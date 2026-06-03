"""Learning augmentations for the OAC solver (Phase 4).

Learning *augments* the optimizer (warm-start initializer, surrogate metric); it does
not replace it. A sibling study (planar-observability-tracking) showed memoryless
learned policies cannot climb the observability margin on their own.
"""
