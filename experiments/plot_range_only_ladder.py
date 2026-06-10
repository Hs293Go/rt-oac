"""Consolidated range-only OAC ladder figure: OA recovers the unobservable (tangential) follower-
position error at every rung (planar -> 3D point-mass -> flat-output [x,y,z,psi]), and the flat-output
OA trajectory transfers to the real quad. Numbers are validated 20-seed medians on the tangential
(motion-observable-only) initial-error direction; bridge = experiments/flatout_bridge.py."""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (rung label, no-OAC final, OAC final, benefit x) -- 20-seed medians, tangential 2.26 m initial error
LADDER = [
    ("Rung 1\nplanar unicycle\n(2D, range-only)", 2.197, 0.186, 11.8),
    ("Rung 2\n3D point-mass\n(range-only)", 2.149, 0.167, 12.9),
    ("Rung 3a\nflat-output [x,y,z,psi]\n(range-only)", 2.250, 0.133, 16.9),
]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))

x = np.arange(len(LADDER))
w = 0.36
noac = [r[1] for r in LADDER]
oac = [r[2] for r in LADDER]
ax[0].bar(x - w / 2, noac, w, color="tab:gray", label="no-OAC (drive straight)")
ax[0].bar(x + w / 2, oac, w, color="tab:red", label="OAC (range-only maneuver)")
for i, r in enumerate(LADDER):
    ax[0].annotate(
        f"{r[3]:.0f}x",
        (i + w / 2, r[2]),
        ha="center",
        va="bottom",
        fontsize=10,
        weight="bold",
    )
    ax[0].annotate(
        f"{r[1]:.2f}", (i - w / 2, r[1]), ha="center", va="bottom", fontsize=8
    )
ax[0].set(
    title="Range-only OAC ladder: OA recovers the unobservable error at every rung\n"
    "(final follower-position error, tangential 2.26 m init; 20-seed medians)",
    xticks=x,
    ylabel="final position error [m]",
    ylim=(0, 2.6),
)
ax[0].set_xticklabels([r[0] for r in LADDER], fontsize=8)
ax[0].legend(fontsize=9, loc="upper right")
ax[0].text(
    0.02,
    0.97,
    "no-OAC stuck ~2.2 m\n(tangential unobservable\nwithout motion)",
    transform=ax[0].transAxes,
    fontsize=8,
    va="top",
    color="dimgray",
)

# Panel 2: the rung-3a bridge -- the flat-output OA trajectory tracked on the 10-state quad
BRIDGE = [("no-OAC\n(straight)", 1.97), ("OAC flat-output\n-> quad (tracked)", 0.33)]
xb = np.arange(2)
ax[1].bar(xb, [b[1] for b in BRIDGE], 0.5, color=["tab:gray", "tab:orange"])
for i, b in enumerate(BRIDGE):
    ax[1].annotate(f"{b[1]:.2f} m", (i, b[1]), ha="center", va="bottom", fontsize=10)
ax[1].annotate(
    "6x",
    (1, 0.33),
    ha="center",
    va="bottom",
    fontsize=12,
    weight="bold",
    xytext=(0, 16),
    textcoords="offset points",
    color="tab:orange",
)
ax[1].set(
    title="Rung 3a bridge: the OA flat-output trajectory transfers to the QUAD\n"
    "(range-only EKF on the quad's actual tracked trajectory)",
    xticks=xb,
    ylabel="follower-position recovery [m]",
    ylim=(0, 2.4),
)
ax[1].set_xticklabels([b[0] for b in BRIDGE], fontsize=9)
ax[1].text(
    0.5,
    0.80,
    "OA planned in flat outputs ->\ngeometric tracker -> 10-state quad;\nthe localization benefit "
    "survives\n(gated by trajectory trackability)",
    transform=ax[1].transAxes,
    fontsize=8,
    ha="center",
    va="top",
    color="dimgray",
)

fig.suptitle(
    "Bottom-up range-only OAC ladder (consolidated): beneficial + stable OA, planar -> "
    "flat-output -> quad. Recipe: softmin-eig + bounded standoff + symmetry-break.",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/home/hs293go/python-scripts/rt-oac/report/figures/range_only_ladder.png"
fig.savefig(out, dpi=120)
print("wrote", out)
