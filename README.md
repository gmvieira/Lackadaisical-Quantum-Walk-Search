# Lackadaisical Quantum-Walk Search

## Associated scientific article

**Limits of Resilience and System-Size Dependence in Lackadaisical Quantum-Walk Search under Broken-Link and Self-Loop Phase Faults**

**Author:** Gabriel Mauricio Oswald Vieira

## 1. Overview

This repository provides two complementary Python implementations of lackadaisical quantum-walk spatial search on a two-dimensional torus subject to broken-link noise and self-loop phase faults.

The implementations support the investigation of:

- Ideal quantum-walk search.
- Dynamic broken-link noise.
- Self-loop phase faults.
- Combined broken-link and self-loop phase faults.
- Long-time resilience.
- System-size dependence of search performance.

Both programs compare the loopless quantum walk with the lackadaisical quantum walk using a self-loop weight of 4/N, where N is the total number of vertices.

## 2. Computational implementations

| File | Computational approach |
|---|---|
| `Matrix_Based.py` | Sparse matrix representation of the quantum operators. |
| `Amplitude_Based.py` | Direct update of quantum amplitudes, without constructing global operator matrices. |

Although based on distinct computational paradigms, both implementations represent the same mathematical model.

The matrix-based implementation explicitly constructs the oracle, weighted coin, phase-fault operator, ideal flip-flop shift, and broken-link correction operator.

The amplitude-based implementation performs the corresponding operations directly on the local quantum amplitudes.

## 3. Mathematical model

The system consists of a periodic two-dimensional square lattice with N = L² vertices and five coin components: up, down, left, right, and self-loop.

Each time step consists of the following operations:

1. Application of the search oracle.
2. Application of the weighted Grover coin.
3. Application of self-loop phase faults.
4. Application of the flip-flop shift under dynamic broken-link noise.

For each broken edge, the corresponding shift transposition is cancelled, leaving both directional components unchanged.

Self-loop phase faults multiply the affected self-loop amplitudes by a complex phase factor.

Both noise configurations are independently sampled at each time step.

## 4. Software requirements

- Python 3
- NumPy
- SciPy
- Matplotlib

SciPy is required only by the matrix-based implementation.

The required libraries can be installed using:

    pip install numpy scipy matplotlib

## 5. Input parameters

Both programs provide an identical input configuration section.

| Parameter | Description | Default value |
|---|---|---|
| `EXPERIMENT` | Preset selection; zero enables custom parameters | 0 |
| `L` | Torus side length | 16 |
| `steps` | Number of time steps | 1000 |
| `runs` | Independent realizations per loop weight | 100 |
| `B_PROBABILITIES` | Broken-link probabilities | [0.01] |
| `p_S` | Self-loop phase-fault probability | 0.01 |
| `theta` | Phase rotation angle in radians | π/2 |
| `SEED` | Random number generator seed | 20261122 |
| `SAVE_FIGURES` | Enable figure export | True |
| `CHECK_NORM` | Enable quantum-state normalization checks | True |
| `Y_SCALE` | Linear or logarithmic vertical scale | linear |

The total number of vertices, self-loop weights, and marked-vertex coordinates are derived automatically from the lattice size.

Eight predefined experimental configurations are also available in both programs.

## 6. Execution

To execute the matrix-based implementation:

    python Matrix_Based.py

To execute the amplitude-based implementation:

    python Amplitude_Based.py

Alternatively, both programs can be executed directly from a Python development environment, such as Spyder.

## 7. Numerical results and graphical output

Both implementations compute the ensemble-averaged success probability at each time step.

The graphical output includes:

- Loopless quantum-walk search (blue curve).
- Lackadaisical quantum-walk search (red curve).
- Uniform reference probability, 1/N (dashed gray line).

Both programs use an identical graphical routine and support linear and logarithmic probability scales.

Figures are exported in PDF and PNG formats when figure saving is enabled.

Output filenames identify the computational method, system size, number of steps, number of realizations, noise probabilities, phase angle, graphical scale, and random seed.

## 8. Cross-validation and reproducibility

The two computational implementations were cross-validated under ideal and noisy configurations.

Both programs employ the same random number generator seed and sampling order, enabling direct numerical comparisons under equivalent stochastic conditions.

Quantum-state normalization is monitored throughout the evolution when the corresponding validation option is enabled.

The agreement between the two implementations provides independent computational support for the numerical results presented in the associated scientific article.

## 9. Citation

Researchers using these implementations are requested to cite the associated scientific article:

Gabriel Mauricio Oswald Vieira. *Limits of Resilience and System-Size Dependence in Lackadaisical Quantum-Walk Search under Broken-Link and Self-Loop Phase Faults.*

The arXiv identifier and publication details will be added when available.
