# Experiments

## Purpose
This document records the reproducible experiments used to study federated learning under IID and non-IID client data, with an additional focus on FedProx and communication cost.

## Experiment matrix
| Experiment | Main question | Script |
|---|---|---|
| Centralized baseline | What is the reference performance without federation? | `experiments/centralized.py` |
| FedAvg IID | How does the federated baseline perform with IID clients? | `experiments/fedavg_iid.py` |
| FedAvg non-IID | How does client heterogeneity affect FedAvg? | `experiments/fedavg_non_iid.py` |
| FedProx non-IID | How does proximal optimization behave under heterogeneity? | `experiments/fedprox_non_iid.py` |
| Communication analysis | How much model-update communication is accumulated? | `experiments/communication_analysis.py` |

## Reproduction
Run from the repository root:

```bash
python -m experiments.centralized
python -m experiments.fedavg_iid
python -m experiments.fedavg_non_iid
python -m experiments.fedprox_non_iid
python -m experiments.communication_analysis
```

## Configuration
- MNIST
- SimpleCNN
- 5 clients
- 5 federated rounds
- 1 local epoch
- SGD optimizer
- Learning rate: 0.01
- Batch size: 64
- Seed: 42 for federated experiments
- Global test accuracy as the primary metric

## Completed results
| Experiment | Final accuracy |
|---|---:|
| FedAvg — IID | 93.95% |
| FedAvg — non-IID | 93.95% |
| FedProx — non-IID | 73.07% |

### Communication
The completed setup records **80.42 MB cumulative communication after 5 rounds**.

### FedProx non-IID progression
| Round | Accuracy | Cumulative communication |
|---:|---:|---:|
| 1 | 25.68% | 16.08 MB |
| 2 | 55.97% | 32.17 MB |
| 3 | 69.06% | 48.25 MB |
| 4 | 70.91% | 64.34 MB |
| 5 | 73.07% | 80.42 MB |

## Research interpretation
The experiments separate three questions:

1. **Federated baseline:** FedAvg under IID data.
2. **Heterogeneity:** FedAvg under non-IID data.
3. **Alternative optimization:** FedProx under non-IID data.

The results should be interpreted only for the stated dataset, model, partitioning strategy, client count, and training budget. They do not establish a universal superiority relationship between FedAvg and FedProx.

## Reproducibility notes
Keep the Python environment and dependency versions consistent when reproducing results. The repository structure separates algorithms, clients, server components, data, experiments, and utilities to make the experimental pipeline easier to inspect.

## Limitations
The current evaluation uses MNIST, a small CNN, five clients, five rounds, and a specific non-IID partition. Communication is estimated from model-update exchange rather than measured network traffic.
