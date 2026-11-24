"""Lackadaisical quantum-walk search on a periodic 2D square lattice.
Sparse matrix implementation: B(t) S F(t) C Q.
Broken edges leave both directional basis states unchanged.
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, eye, kron


# ===================== INPUT PARAMETERS =====================
# Select one preset (1–8). Set EXPERIMENT = 0 for custom parameters.
EXPERIMENT = 0
# Custom parameters (also the default reference experiment).
L = 16                  # Torus side length
steps = 1000            # Number of time steps
runs = 100              # Independent realizations per loop weight
B_PROBABILITIES = [0.01]
p_S = 0.01              # Self-loop phase-fault probability
theta = np.pi / 2       # Phase angle in radians
SEED = 20261122
SAVE_FIGURES = True
CHECK_NORM = True
Y_SCALE = 'linear'      # 'linear' or 'log'

# Presets retained from the original input files:
# (L, steps, runs, p_B, p_S)
PRESETS = {
    1: (16, 600, 100, 0.07, 0.0),
    2: (16, 600, 100, 0.07, 1.0),
    3: (16, 600, 100, 0.07, 0.25),
    4: (16, 600, 100, 0.07, 0.01),
    5: (16, 10000, 50, 0.07, 0.01),
    6: (16, 10000, 50, 0.01, 0.01),
    7: (16, 1000, 50, 0.01, 0.01),
    8: (256, 200, 30, 0.01, 0.0),
}
if EXPERIMENT:
    L, steps, runs, p_B, p_S = PRESETS[EXPERIMENT]
    B_PROBABILITIES = [p_B]
N = L * L
weights = [0.0, 4.0 / N]
wx, wy = L // 2, L // 2  # Marked vertex
assert L >= 3 and steps >= 2 and runs >= 1
assert 0 <= p_S <= 1 and all(0 <= p <= 1 for p in B_PROBABILITIES)
# ============================================================

METHOD = 'matrix'

up, down, left, right, stay = range(5)
DIM = 5 * N


def idx(x, y, c):
    """Flatten (x, y, coin) using periodic boundaries."""
    return 5 * (L * (y % L) + (x % L)) + c


def permutation_matrix(destinations):
    """Construct P such that P maps each basis state to its destination."""
    destinations = np.asarray(destinations, dtype=int)
    assert len(destinations) == DIM
    assert np.array_equal(np.sort(destinations), np.arange(DIM)), 'Invalid permutation'
    return csr_matrix((np.ones(DIM), (destinations, np.arange(DIM))),
                      shape=(DIM, DIM), dtype=complex)
# Each physical edge pairs two distinct directional basis states.
EDGES = []
for y in range(L):
    for x in range(L):
        EDGES.append((idx(x, y, right), idx(x + 1, y, left)))
        EDGES.append((idx(x, y, up), idx(x, y + 1, down)))


def build_shift():
    """Construct the ideal flip-flop shift as disjoint edge transpositions."""
    destinations = np.arange(DIM)
    for a, b in EDGES:
        destinations[a] = b
        destinations[b] = a
    return permutation_matrix(destinations)


S = build_shift()


def build_oracle():
    """Reverse all five amplitudes at the marked vertex."""
    diagonal = np.ones(DIM, dtype=complex)
    for c in range(5):
        diagonal[idx(wx, wy, c)] = -1
    return diags(diagonal, format='csr')


Q = build_oracle()


def build_coin(l):
    """Construct the weighted Grover reflection at every vertex."""
    sc = np.array([1, 1, 1, 1, sqrt(l)], dtype=float) / sqrt(4 + l)
    C_local = 2 * np.outer(sc, sc) - np.eye(5)
    return kron(eye(N, format='csr'), csr_matrix(C_local), format='csr'), sc


def build_phase(p_s, angle, generator):
    """Sample independent vertex faults and rotate only self-loop amplitudes."""
    faults = generator.random(N) < p_s
    diagonal = np.ones(DIM, dtype=complex)
    diagonal[np.arange(N) * 5 + stay] = np.exp(1j * angle * faults)
    return diags(diagonal, format='csr')


def build_broken_links(p_b, generator):
    """Undo the ideal shift transposition at each independently broken edge."""
    destinations = np.arange(DIM)
    broken = generator.random(len(EDGES)) < p_b
    for (a, b), is_broken in zip(EDGES, broken):
        if is_broken:
            destinations[a] = b
            destinations[b] = a
    return permutation_matrix(destinations), int(np.sum(broken))


def initial_state(sc):
    """Prepare the uniform position state and weighted coin state."""
    return np.kron(np.ones(N) / sqrt(N), sc).astype(complex)


def success_probability(psi):
    """Sum squared moduli of all five marked-vertex amplitudes."""
    base = idx(wx, wy, 0)
    return float(np.sum(np.abs(psi[base:base + 5]) ** 2))


def validate_operators():
    """Check permutation structure and limiting broken-link cases."""
    I = eye(DIM, format='csr', dtype=complex)
    assert (S @ S - I).nnz == 0
    assert (Q @ Q - I).nnz == 0
    assert (S.conjugate().transpose() @ S - I).nnz == 0
    for p in [0.0, 0.3, 1.0]:
        B, _ = build_broken_links(p, np.random.default_rng(71))
        assert (B @ B - I).nnz == 0
        assert (B @ S).shape == (DIM, DIM)
    B0, _ = build_broken_links(0.0, np.random.default_rng(71))
    B1, _ = build_broken_links(1.0, np.random.default_rng(71))
    assert (B0 @ S - S).nnz == 0
    assert (B1 @ S - I).nnz == 0
    print('Structural tests of Q, S and B: OK')


def walk(steps, l, p_b, p_s, angle, generator):
    """Evolve one realization and record success probability including t=0."""
    C, sc = build_coin(l)
    psi = initial_state(sc)
    probabilities = [success_probability(psi)]  # t=0
    for t in range(1, steps + 1):
        # Oracle, coin, phase, ideal shift, then broken-edge cancellation.
        psi = Q @ psi
        psi = C @ psi
        F = build_phase(p_s, angle, generator)
        psi = F @ psi
        psi = S @ psi
        B, _ = build_broken_links(p_b, generator)
        psi = B @ psi
        if CHECK_NORM:
            np.testing.assert_allclose(np.vdot(psi, psi).real, 1.0,
                                       rtol=1e-10, atol=1e-10,
                                       err_msg=f'Norm changed at step {t}')
        probabilities.append(success_probability(psi))
    return np.asarray(probabilities)


def run_experiment(p_b, weights_arg, generator):
    """Average independent realizations pointwise for each loop weight."""
    means = []
    deviations = []
    for l in weights_arg:
        samples = np.array([walk(steps, l, p_b, p_S, theta, generator)
                             for _ in range(runs)])
        means.append(samples.mean(axis=0))
        deviations.append(samples.std(axis=0, ddof=1) if runs > 1
                       else np.zeros(steps + 1))
    return means, deviations



def graphics(p_b, means):
    """Plot ensemble-averaged success probabilities using a common style."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, (l, curve) in enumerate(zip(weights, means)):
        label = (rf'$\ell=4/N={l:.5f}$' if np.isclose(l, 4/N)
                 else rf'$\ell={l:.5f}$')
        ax.plot(np.arange(steps + 1), curve, color=['#2166AC', '#C62828'][i],
                linewidth=1.4, label=label)
    ax.axhline(1/N, color='gray', linestyle='--', linewidth=1,
               label='Uniform probability (1/N)')
    ax.set_xlabel('Number of steps', fontsize=24)
    ax.set_ylabel('Success probability', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(0, steps)
    ax.set_yscale(Y_SCALE)
    ax.set_ylim(0 if Y_SCALE == 'linear' else min(1e-4, 0.5/N), 1)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.3)
    ax.legend(loc='upper right', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if SAVE_FIGURES:
        name = (f'{METHOD}_N{N}_T{steps}_M{runs}_B{p_b:.4f}_'
                f'S{p_S:.3f}_theta{np.degrees(theta):.1f}_{Y_SCALE}_seed{SEED}')
        fig.savefig(name + '.pdf', bbox_inches='tight')
        fig.savefig(name + '.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_ideal():
    curve = walk(2, 0.0, 0.0, 0.0, 0.0, np.random.default_rng(12))
    expected = np.array([1 / N, 1 / N, 4 / N])
    np.testing.assert_allclose(curve, expected, rtol=1e-12, atol=1e-12)
    print('Ideal test P(0), P(1), P(2): OK', curve)


if __name__ == '__main__':
    validate_operators()
    test_ideal()
    generator = np.random.default_rng(SEED)
    for p_b in B_PROBABILITIES:
        means, deviations = run_experiment(p_b, weights, generator)
        print(f'B={p_b:.4f}, S={p_S:.4f}, theta={np.degrees(theta):.2f} degrees')
        for l, curve in zip(weights, means):
            print(f'ell={l:.8f}: P(0)={curve[0]:.12f}, '
                  f'P(1)={curve[1]:.12f}, P(2)={curve[2]:.12f}')
        graphics(p_b, means)

