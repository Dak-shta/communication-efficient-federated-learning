# Research & Professor Matching Guide

## Purpose
This guide makes it easy for a research supervisor to understand which research themes this project demonstrates and where the evidence is located in the repository.

## Core research areas
- Federated Learning
- Distributed Machine Learning
- Non-IID / Heterogeneous Data
- Federated Optimization
- Communication-Efficient Machine Learning
- Client Drift
- Privacy-Aware / Privacy-Preserving ML
- Edge AI and Resource-Constrained Learning
- Distributed Deep Learning
- Reproducible ML Experimentation

## Evidence map
| Repository area | Research evidence |
|---|---|
| `algorithms/` | Federated optimization implementations |
| `clients/` | Local client training and distributed participation |
| `server/` | Global aggregation / server-side coordination |
| `experiments/fedavg_iid.py` | IID federated baseline |
| `experiments/fedavg_non_iid.py` | Data heterogeneity / non-IID learning |
| `experiments/fedprox_non_iid.py` | Proximal optimization and client-drift mitigation |
| `experiments/communication_analysis.py` | Communication-cost analysis |
| `research/RESEARCH.md` | Research question, methodology, results, limitations and future work |
| `docs/EXPERIMENTS.md` | Reproduction and quantitative experiment record |

## Strongest Mitacs matching themes
### 1. Federated learning / distributed AI
Direct evidence through multi-client local training, server aggregation, FedAvg and FedProx.

### 2. Communication-efficient ML
Direct evidence through cumulative model-update communication analysis.

### 3. Non-IID learning / federated optimization
Direct evidence through non-IID experiments and comparison of FedAvg with FedProx.

### 4. Edge intelligence
The federated setup maps naturally to distributed devices where raw data remains local and communication is constrained.

### 5. Privacy-aware ML
The project studies the federated setting in which data is kept at clients. **Do not claim formal privacy guarantees** because differential privacy or secure aggregation is not currently implemented.

### 6. Efficient distributed deep learning
The project evaluates training quality against a limited number of communication rounds using a lightweight CNN.

## Keywords for project matching
**Primary:** Federated Learning, Distributed Machine Learning, Non-IID Data, FedAvg, FedProx, Federated Optimization, Communication-Efficient ML, Client Drift.

**Secondary:** Edge AI, Distributed Optimization, Privacy-Preserving ML, Efficient AI, Personalized Federated Learning, Model Compression, Resource-Constrained Learning.

## 20-second professor scan
A supervisor can quickly evaluate the project through this sequence:

**Research question → experimental design → implementation → quantitative results → limitations → future research.**

The project is intentionally documented in this order so that it reads as a research project rather than only as a software implementation.
