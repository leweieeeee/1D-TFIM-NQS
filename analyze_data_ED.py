#Compare VMC C_single / energy to exact diagonalization

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 20
J_OVER_H = [0.1, 0.25, 1.0, 3.0, 6.0]
OUT_ROOT = Path("runs")
ED_CATALOG = Path("ed_catalog.json")
test_jh = 0.25

ed_catalog = json.loads(ED_CATALOG.read_text(encoding="utf-8"))["entries"]

all_data = []
for run_dir in OUT_ROOT.glob("jh*_rmax*"):
    obs_path = run_dir / "observables.json"
    if not obs_path.is_file():
        continue
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    if "C_single" not in obs:
        continue
    jh = float(obs["jh"])
    rm = int(obs["rm"])
    ed = ed_catalog.get(f"{jh:g}", {}).get(str(rm))
    if ed is None:
        print(f"skip missing ED catalog entry jh={jh:g} rm={rm}")
        continue
    C = np.asarray(obs["C_single"], dtype=float)
    C_ed = np.asarray(ed["C_single"], dtype=float)
    if C.shape != C_ed.shape:
        print(f"skip shape mismatch {obs_path}")
        continue
    E_ed = float(ed["E_per_site"])
    delta = C - C_ed
    all_data.append(
        {
            "jh": jh,
            "rm": rm,
            "r_distances": np.asarray(obs.get("r_distances", ed["r_distances"]), dtype=float),
            "E_per_site": float(obs["E_per_site"]),
            "ED_E_per_site": E_ed,
            "dE": float(obs["E_per_site"]) - E_ed,
            "r_dE": (float(obs["E_per_site"]) - E_ed) / abs(E_ed),
            "C": C,
            "C_ed": C_ed,
            "delta_C": delta,
            "RMSE": float(np.sqrt(np.mean(delta**2))),
            "r_RMSE": float(np.sqrt(np.mean(delta**2)) / np.linalg.norm(C_ed)),
            "MAE": float(np.mean(np.abs(delta))),
        }
    )

all_data.sort(key=lambda r: (r["jh"], r["rm"]))
print(f"loaded {len(all_data)} runs with VMC observables + ed_catalog")

# ΔE and RMSE vs r_max
fig, (ax_e, ax_rmse,ax_r_e,ax_r_rmse) = plt.subplots(1, 4,figsize=(10, 4), constrained_layout=True)
for jh in J_OVER_H:
    rows = [r for r in all_data if r["jh"] == jh]
    if not rows:
        continue
    rmax = [r["rm"] for r in rows]
    ax_e.plot(rmax, [r["dE"] for r in rows], "o-", label=f"J/h={jh:g}")
    ax_rmse.plot(rmax, [r["RMSE"] for r in rows], "o-", label=f"J/h={jh:g}")
    ax_r_e.plot(rmax,[r["r_dE"] for r in rows], "o-", label = f"J/h={jh:g}")
    ax_r_rmse.plot(rmax,[r["r_RMSE"] for r in rows], "o-", label=f"J/h={jh:g}")


ax_e.axhline(0, color="k", lw=0.5)
ax_e.set_xlabel("r_max")
ax_e.set_ylabel(r"$\Delta E = E_{\mathrm{VMC}} - E_{\mathrm{ED}}$")
ax_e.set_title("Energy difference")
ax_e.grid(alpha=0.3)
ax_e.legend(fontsize=8)

ax_rmse.set_xlabel("r_max")
ax_rmse.set_ylabel(r"RMSE$[C_{\mathrm{single}}(r)]$" )
ax_rmse.set_title("C_single RMSE vs ED")
ax_rmse.set_yscale("log")
ax_rmse.grid(alpha=0.3)
ax_rmse.legend(fontsize=8)

ax_r_e.axhline(0, color="k", lw=0.5)
ax_r_e.set_xlabel("r_max")
ax_r_e.set_ylabel(r"$\Delta E / |E_{\mathrm{ED}}|$")
ax_r_e.set_title("Relative energy difference")
ax_r_e.grid(alpha=0.3)
ax_r_e.legend(fontsize=8)

ax_r_rmse.set_xlabel("r_max")
ax_r_rmse.set_ylabel(r"RRMSE $= \mathrm{RMSE}/\|C_{\mathrm{ED}}\|_2$")
ax_r_rmse.set_title("C_single RRMSE vs ED")
ax_r_rmse.set_yscale("log")
ax_r_rmse.grid(alpha=0.3)
ax_r_rmse.legend(fontsize=8)


#singular pointwise error
fig2, axes = plt.subplots(
    1,
    len(J_OVER_H),
    figsize=(3.2 * len(J_OVER_H), 3.5),
    sharey=False,
    constrained_layout=True,
)


for ax, jh in zip(axes, J_OVER_H):
    rows = [r for r in all_data if r["jh"] == jh]
    for r in rows:
        ax.plot(
            r["r_distances"],
            r["C"],
            "o-",
            ms=3,
            lw=1,
            label=f"r_max={r['rm']}",
            alpha=0.85,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"J/h = {jh:g}")
    ax.set_xlabel("r")
    ax.grid(alpha=0.3)
    if jh == 0.1:
        ax.set_ylim(None,0.09)
    if jh == 0.25:
        ax.set_ylim(None,0.6)
#axes[0].set_ylabel(r"$\delta C(r) = C_{\mathrm{VMC}} - C_{\mathrm{ED}}$")
axes[-1].legend(fontsize=6, ncol=1, loc="best")
fig2.suptitle("C", y=1.02)

fig3, axes = plt.subplots(
    1,
    len(J_OVER_H),
    figsize=(3.2 * len(J_OVER_H), 3.5),
    sharey=False,
    constrained_layout=True,
)


for ax, jh in zip(axes, J_OVER_H):
    rows = [r for r in all_data if r["jh"] == jh]
    for r in rows:
        ax.plot(
            r["r_distances"],
            r["C_ed"],
            "o-",
            ms=3,
            lw=1,
            label=f"r_max={r['rm']}",
            alpha=0.85,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"J/h = {jh:g}")
    ax.set_xlabel("r")
    ax.grid(alpha=0.3)
    if jh == 0.1:
        ax.set_ylim(None,0.09)
    if jh == 0.25:
        ax.set_ylim(None,0.6)
#axes[0].set_ylabel(r"$\delta C(r) = C_{\mathrm{VMC}} - C_{\mathrm{ED}}$")
axes[-1].legend(fontsize=6, ncol=1, loc="best")
fig3.suptitle("C_ed", y=1.02)


# --- pointwise error vs distance: 5 panels (one per J/h) ---
fig4, axes = plt.subplots(
    1,
    len(J_OVER_H),
    figsize=(3.2 * len(J_OVER_H), 3.5),
    sharey=True,
    constrained_layout=True,
)
if len(J_OVER_H) == 1:
    axes = [axes]

for ax, jh in zip(axes, J_OVER_H):
    rows = [r for r in all_data if r["jh"] == jh]
    for r in rows:
        ax.plot(
            r["r_distances"],
            r["delta_C"],
            "o-",
            ms=3,
            lw=1,
            label=f"r_max={r['rm']}",
            alpha=0.85,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"J/h = {jh:g}")
    ax.set_xlabel("r")
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r"$\delta C(r) = C_{\mathrm{VMC}} - C_{\mathrm{ED}}$")
axes[-1].legend(fontsize=6, ncol=1, loc="best")
fig4.suptitle("Pointwise C_single error vs distance", y=1.02)

# --- summary table ---
print(f"{'jh':>6} {'rm':>4} {'dE':>12} {'RMSE':>12} {'MAE':>12} {'dC(1)':>12}")
for r in all_data:
    print(
        f"{r['jh']:6g} {r['rm']:4d} {r['dE']:12.6e} {r['RMSE']:12.6e} "
        f"{r['MAE']:12.6e} {r['delta_C'][0]:12.6e}"
    )

fig5, axes = plt.subplots(
    2,
    len(J_OVER_H),
    figsize=(3.2 * len(J_OVER_H), 3.5),
    sharey=False,
    constrained_layout=True,
)

rows = [r for r in all_data if r["jh"] == test_jh]
for i, r in enumerate(rows):
    ax = axes[i // 5, i % 5]
    ax.plot(
        r["r_distances"],
        r["C"],
        "o-",
        ms=3,
        lw=1,
        label="VMC",
        alpha=0.85,
    )
    ax.plot(
        r["r_distances"],
        r["C_ed"],
        "o-",
        ms=3,
        lw=1,
        label="ED",
        alpha=0.85,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"rmax = {r['rm']}")
    ax.set_xlabel("r")
    ax.grid(alpha=0.3)
    if test_jh == 0.1:
        ax.set_ylim(None, 0.09)
    if test_jh == 0.25:
        ax.set_ylim(None, 0.6)
axes[-1, -1].legend(fontsize=6, ncol=1, loc="best")
fig5.suptitle(f"VMC vs ED C_single (J/h = {test_jh:g})", y=1.02)

# correlation heatmap

fig6, axes = plt.subplots(1,2)
titles = ["C_VMC","C_ed"]


plt.show()

