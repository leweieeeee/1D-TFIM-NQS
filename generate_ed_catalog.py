#Compute ED ground-state energy and C_single + save to ed_catalog.json
import json
from pathlib import Path

import netket as nk
import numpy as np
from netket.operator.spin import sigmaz
from scipy.sparse.linalg import eigsh

from parallel_run_lr import build_hamiltonian, run_label

N = 20
ALPHA = 1.0
H_FIELD = 1.0
J_OVER_H = [0.1, 0.25, 1.0, 3.0, 6.0]
R_MAX = list(range(1, N // 2 + 1))

OUT_PATH = Path(__file__).resolve().parent / "ed_catalog.json"

hi = nk.hilbert.Spin(s=1 / 2, N=N)
r_distances = np.arange(1, N // 2 + 1)
sz = [sigmaz(hi, i).to_sparse() for i in range(N)]

catalog = {}

for jh in J_OVER_H:
    jh_key = f"{float(jh):g}"
    catalog[jh_key] = {}
    J = jh * H_FIELD
    for rm in R_MAX:
        print(f"ED jh={jh:g} r_max={rm}")
        H = build_hamiltonian(hi, J, H_FIELD, rm)
        eig_vals, eig_vecs = eigsh(H.to_sparse(), k=1, which="SA")
        psi = eig_vecs[:, 0]
        E_per_site = float(eig_vals[0].real) / N

        SC = np.zeros(len(r_distances))
        for k, r in enumerate(r_distances):
            vals = []
            for i in range(N):
                j = (i + r) % N
                matrix = sz[i] @ sz[j]
                val = np.vdot(psi, matrix @ psi) / np.vdot(psi, psi)
                vals.append(val)
            SC[k] = float(np.real(np.mean(vals)))

        catalog[jh_key][str(int(rm))] = {
            "jh": float(jh),
            "rm": int(rm),
            "E_per_site": E_per_site,
            "C_single": SC.tolist(),
            "r_distances": r_distances.tolist(),
            "source": f"generated:{run_label(jh, rm)}",
        }

payload = {
    "N": N,
    "alpha": ALPHA,
    "h": H_FIELD,
    "entries": catalog,
    "n_entries": sum(len(v) for v in catalog.values()),
}

OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"wrote {OUT_PATH} with {payload['n_entries']} entries")
