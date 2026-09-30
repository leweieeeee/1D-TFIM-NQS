import matplotlib.pyplot as plt
import jax
import netket as nk
import nqxpack
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

key = jax.random.key(SEED)
key, subkey = jax.random.split(key)
hi = nk.hilbert.Spin(s=1 / 2, N=N)

spin_configs = jax.random.randint(subkey, shape=(200, N), minval=0, maxval=2) * 2 - 1

vit_module = ViT(
    num_layers=4,
    d_model=60,
    n_heads=10,
    patch_size=2,
    transl_invariant=True,
)

template = vit_module.init(subkey,spin_configs)
H = build_hamiltonian(hi,J_OVER_H,H_FIELD,R_MAX)
blob = Path("starter/trained_vit.msgpack").read_bytes()
params = {"params": from_bytes(template["params"],blob)} 
#outer params takes value from from_bytes, inner params extracts "params" from template and assigns it the value from blob

sampler = nk.sampler.MetropolisHamiltonian(
    hilbert = hi,
    hamiltonian=H,
    n_chains = N_SAMPLES,
    sweep_size = N
)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    variables=params,
    n_samples=N_SAMPLES,
    chunk_size=CHUNK_SIZE
)

print ("running")

E_per_site = float(vstate.expect(H).mean.real) / N
print("Energy per site:", E_per_site)
