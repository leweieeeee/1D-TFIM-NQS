import matplotlib.pyplot as plt
import jax
import netket as nk
import nqxpack
import numpy as np
from netket.driver import VMC_SR
from netket.operator.spin import sigmax, sigmaz
from flax.serialization import from_bytes, to_bytes
from pathlib import Path


from SCTransformer_LR import ViT
from run_lr_preliminary import build_hamiltonian

N = 20
ALPHA = 1.0
H_FIELD = 1.0
J_OVER_H = 6.0
R_MAX = 8
N_SAMPLES = int(4096)
CHUNK_SIZE = int(256/2)
N_ITER = 100
LEARNING_RATE = 0.0075
DIAG_SHIFT = 1e-4
SEED = 0

hi = nk.hilbert.Spin(s=1 / 2, N=N)

H = build_hamiltonian(hi,J_OVER_H,H_FIELD,R_MAX)

print ("running")

vstate = nqxpack.load("starter/6.0-8_trained_vit.nk")

E_per_site = float(vstate.expect(H).mean.real) / N
print("Energy per site:", E_per_site)

exp_z = [float(vstate.expect(sigmaz(hi,i)).mean.real) for i in range (N)]
print("Mean <sigma^z>", np.mean(exp_z))

i, j = 0, 1
zz = float(vstate.expect(sigmaz(hi, i) @ sigmaz(hi, j)).mean.real)
C1 = zz - exp_z[i] * exp_z[j]
print("C(1):", C1)