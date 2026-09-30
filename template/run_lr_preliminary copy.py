"""Preliminary LR-TFIM sweep: J/h × bond-length cutoff → labeled checkpoints under runs/lr_preliminary/."""
import json
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from runtime_parallel import (
    apply_thread_env,
    parallel_train_spec,
    resolve_runtime_config,
)

_RUNTIME = resolve_runtime_config()
if os.environ.get("NETKET_WORKER") != "1":
    apply_thread_env(_RUNTIME["threads_per_process"])

import jax
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from flax.serialization import from_bytes, to_bytes
from netket.driver import VMC_SR
from netket.operator.spin import sigmax, sigmaz

from SCTransformer_LR import ViT

N = 20
ALPHA = 1.0
H_FIELD = 1.0
JH_VALUES = [0.1, 0.25, 1.0, 3.0, 6.0]
R_MAX_FULL = N // 2
R_MAX_VALUES = list(range(1, R_MAX_FULL + 1))
OUT_ROOT = Path("runs/lr_preliminary")

VIT_KW = dict(
    num_layers=4,
    d_model=60,
    n_heads=10,
    patch_size=2,
    transl_invariant=True,
)

N_ITER = 250
N_ITER_RETRY = 500
N_CHAINS = _RUNTIME["n_chains"]
CHUNK_SIZE = _RUNTIME["chunk_size"]
SWEEP_WORKERS = _RUNTIME["sweep_workers"]
LEARNING_RATE = 0.0075
DIAG_SHIFT = 1e-4
PLATEAU_WINDOW = 40
PLATEAU_STD_MAX = 0.02
SEED = 0
HEARTBEAT_SECONDS = max(0, int(os.environ.get("NETKET_HEARTBEAT_SEC", "300")))


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [pid={os.getpid()}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def start_heartbeat(label: str, n_iter: int) -> tuple[threading.Event, float]:
    stop_event = threading.Event()
    started_at = time.monotonic()

    if HEARTBEAT_SECONDS <= 0:
        return stop_event, started_at

    def _heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = format_seconds(time.monotonic() - started_at)
            log(f"{label}: vmc.run still running after {elapsed} (n_iter={n_iter})")

    threading.Thread(
        target=_heartbeat,
        name=f"heartbeat-{label}",
        daemon=True,
    ).start()
    return stop_event, started_at


def run_label(j_over_h: float, r_max: int) -> str:
    return f"jh{j_over_h:g}_rmax{r_max}"


def build_hamiltonian(hilbert, J, h, r_max):
    hamiltonian = sum([-h * sigmax(hilbert, i) for i in range(N)])
    for i in range(N):
        for j in range(i + 1, N):
            r = min(j - i, N - (j - i))
            if r > r_max:
                continue
            Jij = J / (r**ALPHA)
            hamiltonian += -Jij * sigmaz(hilbert, i) @ sigmaz(hilbert, j)
    return hamiltonian


def build_sampler(hilbert, hamiltonian, n_chains, sweep_size=N):
    return nk.sampler.MetropolisHamiltonian(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        n_chains=n_chains,
        sweep_size=sweep_size,
    )


def train_run(
    key,
    j_over_h,
    r_max,
    n_iter,
    ckpt_path=None,
    tag="",
    out_root=OUT_ROOT,
    n_chains=N_CHAINS,
    chunk_size=CHUNK_SIZE,
):
    J = j_over_h * H_FIELD
    h = H_FIELD
    label = run_label(j_over_h, r_max)
    run_dir = out_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_path or run_dir / f"{label}.msgpack"
    train_started_at = time.monotonic()

    log(
        f"start {label}{tag}: J={J} h={h} alpha={ALPHA} r_max={r_max} "
        f"n_iter={n_iter} out_dir={run_dir}"
    )

    key, subkey = jax.random.split(key)
    spin_configs = jax.random.randint(subkey, shape=(200, N), minval=0, maxval=2) * 2 - 1

    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
    hamiltonian = build_hamiltonian(hilbert, J, h, r_max)

    vit_module = ViT(**VIT_KW)
    key, subkey = jax.random.split(key)
    if ckpt_path.is_file():
        template = vit_module.init(subkey, spin_configs)
        blob = ckpt_path.read_bytes()
        params = {"params": from_bytes(template["params"], blob)}
        log(f"{label}: loaded checkpoint {ckpt_path}")
    else:
        params = vit_module.init(subkey, spin_configs)

    sampler = build_sampler(hilbert, hamiltonian, n_chains=n_chains)
    key, subkey = jax.random.split(key)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=vit_module,
        sampler_seed=subkey,
        n_samples=n_chains,
        n_discard_per_chain=0,
        variables=params,
        chunk_size=chunk_size,
    )
    log(
        f"{label}: built MCState with n_chains={n_chains} chunk_size={chunk_size} "
        f"devices={len(jax.devices())}"
    )

    vmc = VMC_SR(
        hamiltonian=hamiltonian,
        optimizer=nk.optimizer.Sgd(learning_rate=LEARNING_RATE),
        diag_shift=DIAG_SHIFT,
        variational_state=vstate,
        mode="complex",
    )
    runtime_log = nk.logging.RuntimeLog()
    heartbeat_stop, vmc_started_at = start_heartbeat(label, n_iter)
    log(f"{label}: entering vmc.run")
    try:
        vmc.run(n_iter=n_iter, out=runtime_log)
    finally:
        heartbeat_stop.set()
    log(f"{label}: vmc.run finished in {format_seconds(time.monotonic() - vmc_started_at)}")

    energy_per_site = np.asarray(runtime_log.data["Energy"]["Mean"].real) / N
    tail = energy_per_site[-PLATEAU_WINDOW:]
    plateau_std = float(np.std(tail))
    plateau_ok = plateau_std <= PLATEAU_STD_MAX

    with ckpt_path.open("wb") as f:
        f.write(to_bytes(vstate.parameters))

    fig_path = run_dir / "training_energy.png"
    plt.figure()
    plt.plot(energy_per_site)
    plt.xlabel("Iterations")
    plt.ylabel("Energy per site")
    plt.title(f"{label}{tag}")
    plt.savefig(fig_path)
    plt.close()

    meta = {
        "label": label,
        "hamiltonian": "lr_tfim_1d",
        "N": N,
        "J": J,
        "h": h,
        "j_over_h": j_over_h,
        "alpha": ALPHA,
        "r_max": r_max,
        "r_max_full": R_MAX_FULL,
        "checkpoint": str(ckpt_path.as_posix()),
        "vit": VIT_KW,
        "vmc": {
            "n_chains": n_chains,
            "chunk_size": chunk_size,
            "sampler": "MetropolisHamiltonian",
            "learning_rate": LEARNING_RATE,
            "diag_shift": DIAG_SHIFT,
            "n_iter": n_iter,
        },
        "energy_plot": str(fig_path.as_posix()),
        "E_per_site_last": float(energy_per_site[-1]),
        "plateau_std_last_window": plateau_std,
        "plateau_ok": plateau_ok,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(
        f"finished {label}{tag}: saved={ckpt_path} E/N={meta['E_per_site_last']:.5f} "
        f"plateau_std={plateau_std:.5f} plateau_ok={plateau_ok} "
        f"elapsed={format_seconds(time.monotonic() - train_started_at)}"
    )
    return meta, key


def _make_spec(j_over_h, r_max, n_iter, seed, ckpt_path=None, tag=""):
    return {
        "j_over_h": j_over_h,
        "r_max": r_max,
        "n_iter": n_iter,
        "seed": seed,
        "ckpt_path": ckpt_path,
        "tag": tag,
        "n_chains": N_CHAINS,
        "chunk_size": CHUNK_SIZE,
        "threads": _RUNTIME["threads_per_process"],
    }


def run_batch(pending, n_iter, by_label, combos, key, ckpt_by_label=None, tag=""):
    """Run a list of (jh, r_max) jobs sequentially or in parallel."""
    ckpt_by_label = ckpt_by_label or {}
    if not pending:
        log(f"{tag or 'pass'}: no pending runs")
        return key

    if SWEEP_WORKERS <= 1:
        log(f"running batch sequentially: {len(pending)} runs")
        for j_over_h, r_max in pending:
            label = run_label(j_over_h, r_max)
            meta, key = train_run(
                key,
                j_over_h,
                r_max,
                n_iter=n_iter,
                ckpt_path=ckpt_by_label.get(label),
                tag=tag,
            )
            by_label[meta["label"]] = meta
            write_manifest(by_label, combos)
            log(f"progress: {len(by_label)}/{len(combos)} runs recorded after {label}")
        return key

    specs = []
    for i, (j_over_h, r_max) in enumerate(pending):
        label = run_label(j_over_h, r_max)
        specs.append(
            _make_spec(
                j_over_h,
                r_max,
                n_iter,
                SEED + i + hash((n_iter, tag, label)) % 10_000,
                ckpt_path=ckpt_by_label.get(label),
                tag=tag,
            )
        )

    mp_ctx = mp.get_context("spawn")
    log(
        f"running batch in parallel: {len(specs)} runs with max_workers={SWEEP_WORKERS} "
        f"start_method={mp_ctx.get_start_method()}"
    )
    with ProcessPoolExecutor(max_workers=SWEEP_WORKERS, mp_context=mp_ctx) as pool:
        futures = {pool.submit(parallel_train_spec, spec): spec for spec in specs}
        for fut in as_completed(futures):
            meta = fut.result()
            by_label[meta["label"]] = meta
            write_manifest(by_label, combos)
            log(
                f"progress: {len(by_label)}/{len(combos)} runs recorded after {meta['label']}"
            )
    return key


def load_completed_runs(out_root=OUT_ROOT):
    """Load meta for runs that already have a checkpoint on disk."""
    by_label = {}
    if not out_root.is_dir():
        return by_label
    for meta_path in sorted(out_root.glob("jh*_rmax*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ckpt = Path(meta["checkpoint"])
        if ckpt.is_file():
            by_label[meta["label"]] = meta
    return by_label


def write_manifest(by_label, combos, out_root=OUT_ROOT):
    manifest = {
        "hamiltonian": "H = -h sum sigma^x - J sum_{i<j, r<=r_max} sigma^z sigma^z / r^alpha (1D PBC)",
        "N": N,
        "r_max_full": R_MAX_FULL,
        "sampler": "MetropolisHamiltonian",
        "runtime": {
            "highmem": _RUNTIME["highmem"],
            "sweep_workers": SWEEP_WORKERS,
            "threads_per_process": _RUNTIME["threads_per_process"],
            "n_chains": N_CHAINS,
            "chunk_size": CHUNK_SIZE,
        },
        "runs": [by_label[run_label(jh, rm)] for jh, rm in combos if run_label(jh, rm) in by_label],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    log(f"JAX devices: {jax.devices()} backend={jax.default_backend()}")
    log(
        f"Runtime: highmem={_RUNTIME['highmem']} sweep_workers={SWEEP_WORKERS} "
        f"threads/process={_RUNTIME['threads_per_process']} "
        f"n_chains={N_CHAINS} chunk_size={CHUNK_SIZE} cpus={_RUNTIME['cpu_count']} "
        f"heartbeat={HEARTBEAT_SECONDS}s"
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    combos = [(jh, rm) for jh in JH_VALUES for rm in R_MAX_VALUES]
    by_label = load_completed_runs(OUT_ROOT)
    pending = [(jh, rm) for jh, rm in combos if run_label(jh, rm) not in by_label]
    log(
        f"Pass 1: {len(pending)}/{len(combos)} remaining, n_iter={N_ITER}, "
        f"jh in {JH_VALUES}"
    )
    if by_label:
        log(f"skipping {len(by_label)} completed: {', '.join(sorted(by_label))}")

    key = jax.random.key(SEED)
    key = run_batch(pending, N_ITER, by_label, combos, key)

    failed = [m["label"] for m in by_label.values() if not m["plateau_ok"]]
    if failed:
        log(
            f"Pass 2 (retry): {len(failed)} runs, +{N_ITER_RETRY} iter from checkpoint "
            f"(sequential)"
        )
        for label in failed:
            m = by_label[label]
            meta, key = train_run(
                key,
                m["j_over_h"],
                m["r_max"],
                n_iter=N_ITER_RETRY,
                ckpt_path=Path(m["checkpoint"]),
                tag=" retry",
            )
            by_label[meta["label"]] = meta
            write_manifest(by_label, combos)

    manifest = write_manifest(by_label, combos)

    ok = sum(1 for m in manifest["runs"] if m["plateau_ok"])
    log(
        f"Wrote {OUT_ROOT / 'manifest.json'} ({len(manifest['runs'])} runs, "
        f"plateau_ok={ok}/{len(manifest['runs'])})"
    )
