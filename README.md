# Communication-Efficient Federated Learning

## Research Project

**Communication-Efficient and Privacy-Aware Federated Learning under Non-IID Data**

This project investigates how data heterogeneity and communication constraints affect federated learning performance, with experiments using FedAvg and FedProx on MNIST.

## Research question
> How do data heterogeneity and communication constraints affect federated learning performance, and how does FedProx behave under non-IID client data compared with FedAvg?

## What is implemented
- Centralized MNIST baseline
- Federated client/server architecture
- FedAvg with IID data
- FedAvg with non-IID data
- FedProx with non-IID data
- Communication-cost analysis
- Modular experiment scripts
- Reproducible research documentation

## Key results
| Experiment | Final accuracy |
|---|---:|
| FedAvg — IID | 93.95% |
| FedAvg — non-IID | 93.95% |
| FedProx — non-IID | 73.07% |

**Cumulative communication:** 80.42 MB after 5 federated rounds in the configured 5-client setup.

## Research areas
- Federated Learning
- Distributed Machine Learning
- Non-IID Learning
- Federated Optimization
- Communication-Efficient ML
- Client Drift
- Privacy-Aware ML
- Edge AI
- Efficient Distributed Deep Learning

## Research documentation
- [`research/RESEARCH.md`](research/RESEARCH.md) — research question, methodology, results, limitations and future work
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — experiment matrix and reproduction guide
- [`docs/RESEARCH_GUIDE.md`](docs/RESEARCH_GUIDE.md) — professor/project matching guide

## Reproduction
From the repository root:

```bash
python -m experiments.centralized
python -m experiments.fedavg_iid
python -m experiments.fedavg_non_iid
python -m experiments.fedprox_non_iid
python -m experiments.communication_analysis
```

## Mitacs relevance
This repository demonstrates hands-on experience with federated learning, distributed optimization, heterogeneous/non-IID data, communication-aware experimentation, PyTorch, quantitative evaluation, and reproducible research documentation.

It is particularly relevant to research projects involving **federated learning, distributed AI, edge intelligence, efficient ML, privacy-aware ML, non-IID optimization, and resource-constrained learning**.

## Future research
Potential extensions include stronger non-IID settings, more clients, partial participation, communication compression, convergence-vs-communication analysis, larger datasets/models, personalized federated learning, and privacy-preserving mechanisms.
