"""Load LR checkpoints; compute <sigma^z> and connected C(r) = <z_i z_j> - <z_i><z_j>."""
import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from flax.serialization import from_bytes
from netket.operator.spin import sigmax, sigmaz

from SCTransformer_LR import ViT

MANIFEST = Path("runs/lr_preliminary/manifest.json")
OUT_ROOT = Path("runs/lr_preliminary")


def build_vstate(meta, chain_mult=1):
    N = meta["N"]
    vit_kw = {
        k: meta["vit"][k]
        for k in ("num_layers", "d_model", "n_heads", "patch_size", "transl_invariant")
    }
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    spin_configs = jax.random.randint(subkey, shape=(200, N), minval=0, maxval=2) * 2 - 1

    vit_module = ViT(**vit_kw)
    template = vit_module.init(subkey, spin_configs)
    blob = Path(meta["checkpoint"]).read_bytes()
    params = {"params": from_bytes(template["params"], blob)}

    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
    hamiltonian = build_hamiltonian(hilbert, meta)
    vmc = meta["vmc"]
    n_chains = vmc["n_chains"] * chain_mult
    chunk_size = vmc["chunk_size"] * chain_mult
    sampler_kind = vmc.get("sampler", "MetropolisExchange")
    if sampler_kind == "MetropolisHamiltonian":
        sampler = nk.sampler.MetropolisHamiltonian(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            n_chains=n_chains,
            sweep_size=N,
        )
    else:
        lattice = nk.graph.Chain(length=N, pbc=True)
        sampler = nk.sampler.MetropolisExchange(
            hilbert=hilbert,
            graph=lattice,
            d_max=vmc["d_max"],
            n_chains=n_chains,
            sweep_size=lattice.n_nodes,
        )
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=n_chains,
        n_discard_per_chain=0,
        variables=params,
        chunk_size=chunk_size,
    )
    return vstate, hilbert


def build_hamiltonian(hilbert, meta):
    N, J, h, alpha, r_max = meta["N"], meta["J"], meta["h"], meta["alpha"], meta["r_max"]
    hamiltonian = sum([-h * sigmax(hilbert, i) for i in range(N)])
    for i in range(N):
        for j in range(i + 1, N):
            r = min(j - i, N - (j - i))
            if r > r_max:
                continue
            Jij = J / (r**alpha)
            hamiltonian += -Jij * sigmaz(hilbert, i) @ sigmaz(hilbert, j)
    return hamiltonian


def connected_zz_by_distance(vstate, hilbert, N):
    """C(r) averaged over all site pairs at PBC distance r."""
    exp_z = np.array(
        [float(vstate.expect(sigmaz(hilbert, i)).mean.real) for i in range(N)]
    )
    r_distances = np.arange(1, N // 2 + 1)
    C = np.zeros(len(r_distances))
    for k, r in enumerate(r_distances):
        vals = []
        for i in range(N):
            j = (i + r) % N
            zz = float(vstate.expect(sigmaz(hilbert, i) @ sigmaz(hilbert, j)).mean.real)
            vals.append(zz - exp_z[i] * exp_z[j])
        C[k] = np.mean(vals)
    return exp_z, r_distances, C


def load_run(label: str, chain_mult=1):
    meta = json.loads((OUT_ROOT / label / "meta.json").read_text(encoding="utf-8"))
    vstate, hilbert = build_vstate(meta, chain_mult=chain_mult)
    hamiltonian = build_hamiltonian(hilbert, meta)
    exp_z, r_distances, C_connected = connected_zz_by_distance(vstate, hilbert, meta["N"])
    E_per_site = float(vstate.expect(hamiltonian).mean.real) / meta["N"]
    return meta, exp_z, r_distances, C_connected, E_per_site


def save_observables(label, meta, exp_z, r_distances, C_connected, E_per_site):
    out = OUT_ROOT / label / "spin_observables.npz"
    np.savez_compressed(
        out,
        exp_z=exp_z,
        r_distances=r_distances,
        C_connected=C_connected,
        N=meta["N"],
        E_per_site=E_per_site,
        j_over_h=meta["j_over_h"],
        r_max=meta["r_max"],
        J=meta["J"],
        h=meta["h"],
        alpha=meta["alpha"],
    )
    return out


def plot_connected(label, meta, r_distances, C_connected):
    fig_path = OUT_ROOT / label / "connected_zz.png"
    plt.figure(figsize=(6, 4))
    plt.plot(r_distances, C_connected, "o-", lw=1.5, ms=5)
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(meta["r_max"], color="gray", ls="--", lw=0.8, label=f"r_max={meta['r_max']}")
    plt.xlabel("distance r")
    plt.ylabel(r"$\langle\sigma^z_i \sigma^z_j\rangle - \langle\sigma^z_i\rangle\langle\sigma^z_j\rangle$")
    plt.title(f"{label}  (J/h={meta['j_over_h']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    return fig_path


def plot_summary(all_results):
    """3 panels: C(r) vs r, one curve per Hamiltonian r_max at each J/h."""
    fig_path = OUT_ROOT / "connected_zz_summary.png"
    jh_vals = sorted({m["j_over_h"] for m, _, _, _ in all_results})
    fig, axes = plt.subplots(1, len(jh_vals), figsize=(5 * len(jh_vals), 4), sharey=True)
    if len(jh_vals) == 1:
        axes = [axes]

    for ax, jh in zip(axes, jh_vals):
        subset = [(m, rd, cc) for m, rd, cc, _ in all_results if m["j_over_h"] == jh]
        subset.sort(key=lambda x: x[0]["r_max"])
        for meta, r_distances, C_connected in subset:
            rm = meta["r_max"]
            ax.plot(
                r_distances,
                C_connected,
                "o-",
                ms=3,
                lw=1,
                label=f"r_max={rm}",
                alpha=0.85 if rm in (1, meta["r_max_full"]) else 0.45,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("distance r")
        ax.set_title(f"J/h = {jh}")
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    axes[0].set_ylabel(r"connected $C(r)$")
    fig.suptitle("Connected zz correlator vs distance", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_C1_heatmap(all_results):
    """Heatmap of C(1) vs j/h and Hamiltonian r_max."""
    jh_vals = sorted({m["j_over_h"] for m, _, _, _ in all_results})
    rmax_vals = sorted({m["r_max"] for m, _, _, _ in all_results})
    M = np.full((len(jh_vals), len(rmax_vals)), np.nan)
    for meta, _, C, _ in all_results:
        i = jh_vals.index(meta["j_over_h"])
        j = rmax_vals.index(meta["r_max"])
        M[i, j] = C[0]

    fig_path = OUT_ROOT / "connected_C1_heatmap.png"
    plt.figure(figsize=(8, 3))
    im = plt.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=0.6)
    plt.colorbar(im, label=r"$C(1)$ connected")
    plt.xticks(range(len(rmax_vals)), rmax_vals)
    plt.yticks(range(len(jh_vals)), [str(jh) for jh in jh_vals])
    plt.xlabel("Hamiltonian r_max")
    plt.ylabel("J/h")
    plt.title("Nearest-neighbor connected correlator")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130)
    plt.close()
    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, help="single run, e.g. jh3_rmax1")
    parser.add_argument("--all", action="store_true", help="all runs in manifest")
    parser.add_argument(
        "--chain-mult",
        type=int,
        default=1,
        help="multiply training n_chains/chunk_size (e.g. 2 for 2x MC samples)",
    )
    args = parser.parse_args()

    if args.all:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        labels = [r["label"] for r in manifest["runs"]]
        print(
            f"Connected observables for {len(labels)} runs "
            f"(chain_mult={args.chain_mult})",
            flush=True,
        )
        all_results = []
        for i, label in enumerate(labels, 1):
            meta, exp_z, r_distances, C_connected, E_per_site = load_run(
                label, chain_mult=args.chain_mult
            )
            npz = save_observables(label, meta, exp_z, r_distances, C_connected, E_per_site)
            fig = plot_connected(label, meta, r_distances, C_connected)
            all_results.append((meta, r_distances, C_connected, E_per_site))
            print(
                f"[{i}/{len(labels)}] {label}  E/N={E_per_site:.4f}  "
                f"C(1)={C_connected[0]:+.3f}  C(5)={C_connected[4]:+.3f}  -> {fig}",
                flush=True,
            )
        summary = plot_summary(all_results)
        heatmap = plot_C1_heatmap(all_results)
        print(f"summary: {summary}", flush=True)
        print(f"heatmap: {heatmap}", flush=True)
        print("done", flush=True)
    else:
        label = args.label or "jh3_rmax1"
        meta, exp_z, r_distances, C_connected, E_per_site = load_run(
            label, chain_mult=args.chain_mult
        )
        npz = save_observables(label, meta, exp_z, r_distances, C_connected, E_per_site)
        fig = plot_connected(label, meta, r_distances, C_connected)
        print(
            f"label={label}  J/h={meta['j_over_h']}  r_max={meta['r_max']}  E/N={E_per_site:.6f}"
        )
        print(f"C(r): {list(zip(r_distances, np.round(C_connected, 4)))}")
        print(f"saved {npz}  {fig}")
