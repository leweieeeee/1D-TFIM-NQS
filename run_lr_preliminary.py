"""
LR-TFIM ground-state VMC with ViT.

Follows NetKet ViT tutorial (ground-state optimization section):
  https://netket.readthedocs.io/en/stable/tutorials/ViT-wave-function.html
"""
import matplotlib.pyplot as plt
import jax
import netket as nk
from netket.driver import VMC_SR
from netket.operator.spin import sigmax, sigmaz

from SCTransformer_LR import ViT

N = 20
ALPHA = 1.0
H_FIELD = 1.0
J_OVER_H = 1.0
R_MAX = 10
N_SAMPLES = 4096
CHUNK_SIZE = 512
N_ITER = 800
LEARNING_RATE = 0.0075
DIAG_SHIFT = 1e-4
SEED = 0


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


if __name__ == "__main__":
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
        n_discard_per_chain=0,
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
    vmc.run(n_iter=N_ITER, out=log)

    energy_per_site = log.data["Energy"]["Mean"].real / N
    print("Last energy per site:", float(energy_per_site[-1]))

    plt.plot(energy_per_site)
    plt.xlabel("Iterations")
    plt.ylabel("Energy per site")
    plt.savefig("training_energy.png")
    plt.close()
