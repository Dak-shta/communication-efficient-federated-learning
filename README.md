# Federated Learning Research Log

## Project

Communication-Efficient and Privacy-Aware Federated Learning under Non-IID Data

## Research Question

How do data heterogeneity and communication constraints affect
federated learning performance?

---

## Paper 1

### Communication-Efficient Learning of Deep Networks
from Decentralized Data

Authors:
McMahan et al.

Year:
2017

### Problem

Traditional centralized machine learning requires collecting
training data in one location. This can be problematic when data
is private, large, or naturally distributed across devices.

### Key Idea

Federated learning keeps training data on local clients.
Clients train locally and send model updates to a central server,
which aggregates them to produce a global model.

### Algorithm

FedAvg — Federated Averaging

### Important Concepts

- Client
- Server
- Local training
- Global model
- Communication round
- Model aggregation
- IID data
- Non-IID data
- Communication efficiency

### Research Question I Want To Investigate

How does FedAvg behave when client datasets become increasingly
non-IID?

### Planned Experiments

1. Centralized training
2. FedAvg with IID data
3. FedAvg with non-IID data
4. FedAvg vs FedProx
5. Communication-cost analysis