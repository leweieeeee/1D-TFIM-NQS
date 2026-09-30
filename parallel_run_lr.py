"""
LR-TFIM ground-state VMC with a vision transformer

Follows NetKet ViT tutorial @
https://netket.readthedocs.io/en/stable/tutorials/ViT-wave-function.html
"""
import matplotlib.pyplot as plt
import jax
import netket as nk
from netket.driver import VMC_SR
from netket.operator.spin import sigmax, sigmaz
from flax.serialization import from_bytes, to_bytes
from SCTransformer_LR import ViT
from pathlib import Path
import json
import nqxpack
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

N = 20
ALPHA = 1.0
H_FIELD = 1.0
#J_OVER_H = [0.1, 0.25, 1.0, 3.0, 6.0]
J_OVER_H = [0.1,0.25,1.0]
R_MAX = list(range(1, N // 2 + 1))
N_SAMPLES = int(4096/2)
CHUNK_SIZE = int(256/2)
N_ITER = 300
LEARNING_RATE = 0.0075
DIAG_SHIFT = 1e-4
SEED = 0
n_discard_per_chain = 0

VAR_TOL = 0.15
MIN_ITER = 200

def run_label(jh, r_max):
    return f"jh{jh:g}_rmax{r_max}"

def stop_on_variance(step, logged_data, driver):
    stats = logged_data.get("Energy")
    if stats is None or step < MIN_ITER:
        return True
    var = float(np.asarray(stats.variance))
    if var < VAR_TOL:
        print(f"stop at step {step}: variance={var:.4g} < {VAR_TOL}")
        return False
    return True

def build_hamiltonian(hi, J, h, r_max):
    """LR-TFIM on a 1D ring: -h sum sigma^x - sum_{i<j, r<=r_max} (J/r^alpha) sigma^z sigma^z."""
    hamiltonian = sum([-h * sigmax(hi, i) for i in range(N)])
    for i in range(N):
        for j in range(i + 1, N):
            r = min(j - i, N - (j - i))
            if r > r_max:
                continue
            Jij = J / (r**ALPHA)
            hamiltonian += -Jij * sigmaz(hi, i) @ sigmaz(hi, j)
    return hamiltonian

OUT_ROOT = Path("runs")


def train_one(J_OVER_H, R_MAX, SEED, out_dir):
    
    print(str(out_dir), " starting")
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(SEED)
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    J = J_OVER_H * H_FIELD
    hamiltonian = build_hamiltonian(hi, J, H_FIELD, R_MAX)

    vit_module = ViT(
        num_layers=4,
        d_model=60,
        n_heads=10,
        patch_size=2,
        transl_invariant=True,
    )

    key, subkey = jax.random.split(key)
    init_spins = jax.random.randint(subkey, shape=(1, N), minval=0, maxval=2) * 2 - 1
    params = vit_module.init(subkey, init_spins)

    sampler = nk.sampler.MetropolisHamiltonian(
        hilbert=hi,
        hamiltonian=hamiltonian,
        n_chains=N_SAMPLES,
        sweep_size=N,
    )

    key, subkey = jax.random.split(key)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=N_SAMPLES,
        n_discard_per_chain=n_discard_per_chain,
        variables=params,
        chunk_size=CHUNK_SIZE,
    )

    vmc = VMC_SR(
        hamiltonian=hamiltonian,
        optimizer=nk.optimizer.Sgd(learning_rate=LEARNING_RATE),
        diag_shift=DIAG_SHIFT,
        variational_state=vstate,
        mode="complex",
    )

    log = nk.logging.RuntimeLog()
    vmc.run(n_iter=N_ITER, out=log, callback=stop_on_variance)
    #save as nk
    nk_path = out_dir/"state.nk"
    nqxpack.save(vstate, nk_path)

    energy_per_site = log.data["Energy"]["Mean"].real / N   
    #save as npy
    curve_path = out_dir / "energy_data.npy"
    np.save(curve_path, np.asarray(energy_per_site))
    #save as png
    fig_path = out_dir / "energy_plot.png"
    plt.figure()
    plt.plot(energy_per_site)
    plt.xlabel("Iteration")
    plt.ylabel("E / N")
    plt.title(str(out_dir))
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close() 
    #save as meta.json
    meta = {
        "label": str(out_dir),
        "j_over_h": J_OVER_H,
        "r_max": R_MAX,
        "seed": SEED,
        "n_iter": N_ITER,
        "n_samples": N_SAMPLES,
        "learning_rate":LEARNING_RATE,
        "diag_shift":DIAG_SHIFT,
        "checkpoint": str(nk_path),
        "E_per_site_last": float(energy_per_site[-1]),
        "energy_curve": str(curve_path),
        "energy_plot": str(fig_path)
    }
    
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta

def write_manifest(by_label, out_root=OUT_ROOT):
    manifest = {
        "N": N,
        "alpha": ALPHA,
        "h": H_FIELD,
        "j_over_h_grid": list(J_OVER_H),
        "r_max_grid": list(R_MAX),
        "runs": [by_label[k] for k in sorted(by_label)],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest

def worker(spec):
    return train_one(
        spec["j_over_h"],
        spec["r_max"],
        spec["seed"],
        Path(spec["out_dir"]),
    )

def run_sweep(j_over_h, r_max, max_workers, threads_per_process):
    specs = []
    i = 0
    for jh in j_over_h:
        for rm in r_max:
            label = run_label(jh, rm)
            specs.append({
                "j_over_h": jh,
                "r_max": rm,
                "seed": SEED + i,
                "out_dir": str(OUT_ROOT / label),
                "threads": threads_per_process,
            })
            i += 1

    ctx = mp.get_context("spawn")
    by_label = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {pool.submit(worker, s): s for s in specs}
        for fut in as_completed(futures):
            meta = fut.result()
            label = run_label(meta["j_over_h"], meta["r_max"])
            by_label[label] = meta
            write_manifest(by_label)
            print(f"done {label}: E/N={float(meta['E_per_site_last']):.6f}")
    return by_label
    
if __name__ == "__main__":
    cpus = os.cpu_count() or 4
    max_workers = 4
    threads_per_process = max(1, cpus // max_workers)

    os.environ["OMP_NUM_THREADS"] = str(threads_per_process)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_process)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_process)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_process)
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

    run_sweep(J_OVER_H, R_MAX, max_workers, threads_per_process)