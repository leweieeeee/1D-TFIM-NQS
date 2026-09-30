#sigmaZ, energy, connected correlator

from re import T
import matplotlib.pyplot as plt
import jax
import netket as nk
import nqxpack
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from pathlib import Path
import json

from parallel_run_lr import build_hamiltonian, run_label

N = 20
ALPHA = 1.0
H_FIELD = 1.0
J_OVER_H = [0.1, 0.25, 1.0, 3.0, 6.0]
R_MAX = list(range(1, N // 2 + 1))
N_SAMPLES = int(4096/2)
CHUNK_SIZE = int(256/2)
N_ITER = 200
LEARNING_RATE = 0.0075
DIAG_SHIFT = 1e-4
SEED = 0
REHEAT_CHAINS = False

hi = nk.hilbert.Spin(s=1 / 2, N=N)
OUT_ROOT = Path("runs")
all_data = []
 
for obs_path in OUT_ROOT.glob("jh*_rmax*"):
    if REHEAT_CHAINS: 
        obs_path = obs_path / "reheat_observables.json"
    else:
        obs_path =  obs_path / "observables.json"
    obs = json.loads(obs_path.read_text(encoding = "utf-8"))
    all_data.append({
        "jh": obs["jh"],
        "rm": obs["rm"],
        "E_per_site":obs["E_per_site"],
        "exp_z":obs["exp_z"],
        "C_connected":obs["C_connected"],
        "C1":obs["C1"],
        "r_distances":obs["r_distances"]
        })

all_data.sort(key=lambda r: (r["jh"], r["rm"]))


def add_line(ax, data, jh):
    ax.plot(R_MAX, data, label=f"jh = {jh}")

fig = plt.figure(figsize=(16, 4), constrained_layout=True)
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)

for jh in J_OVER_H:
    E_per_site = [row["E_per_site"] for row in all_data if row["jh"] == jh]
    mean_exp_z = [np.mean(row["exp_z"]) for row in all_data if row["jh"] == jh]
    abs_mean_exp_z = [np.mean(np.abs(row["exp_z"])) for row in all_data if row["jh"] == jh]
    C1 = [row["C1"] for row in all_data if row["jh"] == jh]

    add_line(ax1, E_per_site, jh)
    add_line(ax2, mean_exp_z, jh)
    add_line(ax3, abs_mean_exp_z, jh)
    add_line(ax4, C1, jh)

for ax, ylabel in (
    (ax1, "Energy per site"),
    (ax2, "mean_exp_z"),
    (ax3, "|mean_exp_z|"),
    (ax4, "C1"),
):
    ax.set_title(ylabel)
    ax.set_xlabel("r_max")
    ax.set_ylabel(ylabel)
    ax.set_yscale("symlog")
    ax.grid(alpha=0.3)
    ax.legend()


rows = []

for jh in (J_OVER_H):
    mean_exp_z = np.array([np.mean(row["exp_z"]) for row in all_data if row["jh"]==jh])
    C_connected = np.array([np.mean(row["C_connected"]) for row in all_data if row["jh"]==jh])

    rows.append([mean_exp_z,C_connected])

values = np.array(rows)

fig, axes = plt.subplots(1,3)
titles = ["exp_z", "C_connected","abs_exp_z"]

for i, ax in enumerate(axes):
    if i == 2:
        tvals = values[:,i-2,:]
        tvals = np.absolute(tvals)
    else:
        tvals = values[:,i,:]
    im = ax.imshow(tvals)
    ax.set_title(f"{titles[i]}")

    ax.set_xticks(range(len(R_MAX)))
    ax.set_xticklabels(R_MAX)
    ax.set_yticks(range(len(J_OVER_H)))
    ax.set_yticklabels(J_OVER_H)

    ax.set_xlabel("r_max")
    ax.set_ylabel("J/h")
fig.tight_layout()


fig, axes = plt.subplots(1, len(J_OVER_H), figsize=(3 * len(J_OVER_H), 3),
                         sharey=True, constrained_layout=True)
for ax, jh in zip(axes, J_OVER_H):
    for row in all_data:
        if row["jh"] != jh:
            continue
        ax.plot(row["r_distances"], row["C_connected"], label=f"rmax={row['rm']}")
    ax.set_title(f"J/h = {jh}")
    ax.set_xlabel("r")
axes[0].set_ylabel("C_connected")
axes[-1].legend(fontsize=7)

fig.suptitle("CC vs site")


fig, axes = plt.subplots(1, len(J_OVER_H), figsize=(3 * len(J_OVER_H), 3),
                         sharey=True, constrained_layout=True)
sites = np.arange(N)
for ax, jh in zip(axes, J_OVER_H):
    for row in all_data:
        if row["jh"] != jh:
            continue
        ax.plot(sites, row["exp_z"], label=f"rmax={row['rm']}")
    ax.set_title(f"J/h = {jh}")
    ax.set_xlabel("site")
axes[0].set_ylabel("exp_z")
axes[-1].legend(fontsize=7)

fig.suptitle("exp_z vs site")
plt.show()


"""
jh:
    mean exp_z
    c1
    e_per_site
rmax 1-10:
    c connected
    exp_z

  H = build_hamiltonian(hi, jh * H_FIELD, H_FIELD, rm)
"""

