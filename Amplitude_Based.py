"""Lackadaisical quantum-walk search on a periodic 2D square lattice.
Direct amplitude-update implementation: modified flip-flop shift after oracle, coin and phase.
Broken edges leave both directional basis states unchanged.
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

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

METHOD = 'amplitude'

up, down, left, right, stay = range(5)
# Each physical edge pairs two distinct directional basis states.
EDGES = []
for y in range(L):
    for x in range(L):
        EDGES.append(((x, y, right), ((x + 1) % L, y, left)))
        EDGES.append(((x, y, up), (x, (y + 1) % L, down)))
assert len(EDGES) == 2 * N


def initial_state(l):
    """Prepare the uniform position state and weighted coin state."""
    sc = np.array([1., 1., 1., 1., sqrt(l)], dtype=complex) / sqrt(4 + l)
    psi = np.broadcast_to(sc / sqrt(N), (L, L, 5)).copy()
    return psi


def sample_broken_links(p_b, generator):
    """Sample one independent Bernoulli variable per physical edge."""
    return generator.random(2 * N) < p_b


def sample_phase_faults(probability, generator):
    """Sample independent vertex faults in row-major order."""
    return generator.random(N).reshape(L, L) < probability


def flip_flop_shift(tmp, broken):
    """Exchange amplitudes on intact edges; leave broken-edge components unchanged."""
    if len(broken) != 2 * N:
        raise ValueError('Esperados 2N indicadores de arestas broken')
    result = np.empty_like(tmp)
    result[:, :, stay] = tmp[:, :, stay]
    for i, (a, b) in enumerate(EDGES):
        xa, ya, ca = a
        xb, yb, cb = b
        if broken[i]:
            result[ya, xa, ca] = tmp[ya, xa, ca]
            result[yb, xb, cb] = tmp[yb, xb, cb]
        else:
            result[ya, xa, ca] = tmp[yb, xb, cb]
            result[yb, xb, cb] = tmp[ya, xa, ca]
    return result


def success_probability(psi):
    """Sum squared moduli of all five marked-vertex amplitudes."""
    return float(np.sum(np.abs(psi[wy, wx, :]) ** 2))


def step(psi, l, phase_faults, broken, angle):
    """Apply oracle, weighted coin, self-loop phase and modified flip-flop shift."""
    psi = psi.copy()
    # Oracle: reverse every coin component at the marked vertex.
    psi[wy, wx, :] *= -1
    # Weighted Grover reflection without a global matrix.
    weighted_sum = (psi[:, :, up] + psi[:, :, down] +
            psi[:, :, left] + psi[:, :, right] +
            sqrt(l) * psi[:, :, stay])
    weighted_mean = weighted_sum / (4 + l)
    tmp = -psi
    tmp[:, :, :4] += 2 * weighted_mean[:, :, None]
    tmp[:, :, stay] += 2 * sqrt(l) * weighted_mean
    # Phase rotation acts exclusively on the self-loop component.
    tmp[:, :, stay] *= np.where(phase_faults, np.exp(1j * angle), 1.0)
    return flip_flop_shift(tmp, broken)


def walk(steps, l, p_b, p_s, angle, generator):
    """Evolve one realization and record success probability including t=0."""
    psi = initial_state(l)
    curve = np.empty(steps + 1)
    curve[0] = success_probability(psi)
    for t in range(1, steps + 1):
        faults = sample_phase_faults(p_s, generator)
        broken = sample_broken_links(p_b, generator)
        psi = step(psi, l, faults, broken, angle)
        if CHECK_NORM:
            np.testing.assert_allclose(np.vdot(psi, psi).real, 1.,
                                       atol=1e-10, rtol=1e-10,
                                       err_msg=f'Norm changed at t={t}')
        curve[t] = success_probability(psi)
    peak_step = 1 + int(np.argmax(curve[1:]))
    cost = peak_step / sqrt(curve[peak_step]) if curve[peak_step] > 0 else float('inf')
    return curve, cost


def run_experiment(p_b, weights_arg, generator):
    """Average independent realizations pointwise for each loop weight."""
    means, deviations, costs = [], [], []
    for l in weights_arg:
        curves, run_costs = [], []
        for _ in range(runs):
            curve, cost = walk(steps, l, p_b, p_S, theta, generator)
            curves.append(curve)
            run_costs.append(cost)
        curves = np.asarray(curves)
        means.append(curves.mean(axis=0))
        deviations.append(curves.std(axis=0, ddof=1) if runs > 1 else np.zeros(steps + 1))
        costs.append(float(np.mean(run_costs)))
    return means, deviations, costs



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
    generator = np.random.default_rng(12)
    psi = initial_state(0.0)
    probs = [success_probability(psi)]
    for _ in range(2):
        psi = step(psi, 0.0, np.zeros((L,L), dtype=bool),
                    np.zeros(2*N, dtype=bool), 0.0)
        probs.append(success_probability(psi))
    np.testing.assert_allclose(probs, [1/N, 1/N, 4/N], atol=1e-12, rtol=1e-12)
    print('Ideal test P(0), P(1), P(2): OK', probs)


if __name__ == '__main__':
    test_ideal()
    generator = np.random.default_rng(SEED)
    for p_b in B_PROBABILITIES:
        means, deviations, costs = run_experiment(p_b, weights, generator)
        print(f'B={p_b:.4f}; S={p_S:.4f}; theta={np.degrees(theta):.2f} degrees; seed={SEED}')
        for l, curve in zip(weights, means):
            print(f'ell={l:.8f}: P(0)={curve[0]:.12f}, '
                  f'P(1)={curve[1]:.12f}, P(2)={curve[2]:.12f}')
        graphics(p_b, means)
