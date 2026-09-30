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
J_OVER_H = [0.1, 0.25,1.0]
#J_OVER_H = [0.1, 0.25, 1.0, 3.0, 6.0]
R_MAX = list(range(1, N // 2 + 1))
N_SAMPLES = int(4096)
CHUNK_SIZE = int(256)
DIAG_SHIFT = 1e-4
SEED = 0

hi = nk.hilbert.Spin(s=1 / 2, N=N)
OUT_ROOT = Path("runs")

for jh in J_OVER_H:
    for rm in R_MAX:
        out_dir = OUT_ROOT/run_label(jh,rm)
        H = build_hamiltonian(hi, jh * H_FIELD, H_FIELD, rm)
        print(f"running jh = {jh} r_max = {rm}")
        vstate = nqxpack.load(out_dir / "state.nk")
        vstate.n_discard_per_chain = 50
        vstate.reset()
        vstate.sample()
        vstate.n_discard_per_chain = 0
        E_per_site = float(vstate.expect(H).mean.real) / N

        exp_z = np.array([float(vstate.expect(sigmaz(hi,i)).mean.real) for i in range (N)])
        r_distances = np.arange(1, N // 2 + 1)
        C = np.zeros(len(r_distances))
        SC = np.zeros(len(r_distances))

        for k, r in enumerate(r_distances):
            v1 = []
            v2 = []
            for i in range(N):
                j = (i + r) % N
                zz = float(vstate.expect(sigmaz(hi, i) @ sigmaz(hi, j)).mean.real)
                v1.append(zz - exp_z[i] * exp_z[j])
                v2.append(zz)

            C[k] = np.mean(v1)
            SC[k] = np.mean(v2)

        meta = {
            "E_per_site": E_per_site,
            "exp_z": exp_z.tolist(),
            "r_distances": r_distances.tolist(),
            "C_connected": C.tolist(), 
            "C1": float(C[0]),
            "jh": jh,
            "rm": rm,
            "C_single": SC.tolist()
        }

        (out_dir / "reheat_observables.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
