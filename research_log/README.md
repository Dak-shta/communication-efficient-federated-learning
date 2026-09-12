# Communication-Efficient and Privacy-Aware Federated Learning under Non-IID Data

An empirical study of federated learning under IID and non-IID data
distributions, with a comparison of FedAvg and FedProx and an analysis
of communication cost.

## Research Question

How do data heterogeneity and communication constraints affect
federated learning, and what is the trade-off between model accuracy,
communication cost, and privacy?

## Motivation

Federated learning enables multiple clients to collaboratively train
a machine learning model without directly sharing their raw training
data.

However, real-world federated systems face important challenges:

- Client data may be highly heterogeneous.
- Communication between clients and the server can be expensive.
- Local models may drift under non-IID data.
- Different optimization strategies may behave differently under
  heterogeneous client distributions.

This project experimentally investigates these challenges using MNIST.

## Objectives

1. Establish a centralized learning baseline.
2. Implement federated learning using FedAvg.
3. Compare IID and non-IID client distributions.
4. Evaluate FedProx under non-IID data.
5. Measure communication cost across federated rounds.
6. Analyze failure cases and experimental limitations.

## System Architecture

```text
                 ┌──────────────────┐
                 │   Central Server │
                 │                  │
                 │ Global CNN Model │
                 └────────┬─────────┘
                          │
             Global model distribution
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Client 1 │    │ Client 2 │    │ Client N │
    │          │    │          │    │          │
    │ Local    │    │ Local    │    │ Local    │
    │ Training │    │ Training │    │ Training │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ↓
                  Model aggregation
                         ↓
                  Updated global model
## Experiment 1: FedAvg under IID Data

### Setup

- Dataset: MNIST
- Number of clients: 5
- Data distribution: IID
- Algorithm: FedAvg
- Local epochs: 1
- Communication rounds: 5
- Optimizer: SGD
- Learning rate: 0.01
- Batch size: 64

### Results

| Round | Global Test Accuracy |
|------:|---------------------:|
| 1 | 84.71% |
| 2 | 88.94% |
| 3 | 92.52% |
| 4 | 93.75% |
| 5 | 94.28% |

### Observation

The global model accuracy increased consistently across
communication rounds, indicating that repeated local training
and FedAvg aggregation successfully improved the shared global
model under IID client data.

### Next Experiment

Investigate the effect of non-IID data distributions on FedAvg.

## Experiment 2: FedAvg under Non-IID Data

### Setup

- Dataset: MNIST
- Number of clients: 5
- Data distribution: Non-IID
- Client 1: classes 0, 1
- Client 2: classes 2, 3
- Client 3: classes 4, 5
- Client 4: classes 6, 7
- Client 5: classes 8, 9
- Algorithm: FedAvg
- Local epochs: 1
- Communication rounds: 5
- Optimizer: SGD
- Learning rate: 0.01
- Batch size: 64

### Results

| Round | Global Test Accuracy |
|------:|---------------------:|
| 1 | 32.35% |
| 2 | 62.57% |
| 3 | 72.17% |
| 4 | 75.19% |
| 5 | 76.54% |

### Observation

FedAvg achieved substantially lower global accuracy under the
non-IID data distribution compared with the IID experiment.

The global accuracy increased from 32.35% to 76.54% over five
communication rounds, but remained below the 94.28% obtained
under IID data.

### Interpretation

The result suggests that heterogeneous local data distributions
make federated optimization more difficult. Each client trains
primarily on a limited subset of classes, causing local models
to become specialized toward their local data.

### Next Question

Can FedProx improve the robustness of federated optimization
when clients have heterogeneous data distributions?

## Experiment 3: FedProx under Non-IID Data

### Setup

- Dataset: MNIST
- Number of clients: 5
- Data distribution: Non-IID
- Algorithm: FedProx
- Local epochs: 1
- Communication rounds: 5
- Optimizer: SGD
- Learning rate: 0.01
- Batch size: 64
- Proximal coefficient (μ): 0.01

### Results

| Round | Global Test Accuracy |
|------:|---------------------:|
| 1 | 43.13% |
| 2 | 62.65% |
| 3 | 67.80% |
| 4 | 69.84% |
| 5 | 72.13% |

### Comparison with FedAvg

FedAvg final accuracy: 76.54%

FedProx final accuracy: 72.13%

Difference: 4.41 percentage points.

### Observation

FedProx achieved higher accuracy than FedAvg during the first
communication round, but FedAvg achieved higher accuracy in the
later rounds and finished with a higher final accuracy.

Under this experimental configuration, FedProx did not outperform
FedAvg after five communication rounds.

### Interpretation

The result demonstrates that the effectiveness of a federated
optimization method can depend on the data distribution, model,
training configuration, proximal coefficient, and number of
communication rounds.

Further experiments with different values of μ and longer
training could be used to investigate this behavior.


## Experiment 5 — Communication Cost

The model size was measured from the transmitted model parameters.

Experimental configuration:
- Clients: 5
- Full client participation
- Server-to-client transmission
- Client-to-server transmission

Cumulative communication:

| Round | Communication |
|------:|--------------:|
| 1 | 16.08 MB |
| 2 | 32.17 MB |
| 3 | 48.25 MB |
| 4 | 64.34 MB |
| 5 | 80.42 MB |

Observation:

Communication cost increased approximately linearly with the number
of federated rounds because the same model was transmitted to and
from all participating clients during each round.
---

## Limitations

1. The experiments use MNIST rather than a large-scale real-world
   federated dataset.

2. Client participation is assumed to be complete in every round.

3. The non-IID distribution uses a controlled class-skewed partition
   rather than naturally occurring client distributions.

4. Only five communication rounds were evaluated.

5. The CNN architecture is intentionally lightweight and does not
   represent the communication cost of larger production models.

6. Differential privacy has not yet been experimentally evaluated.

7. The experiments do not model network latency, packet loss,
   bandwidth variation, or client availability.

8. Results may depend on hyperparameters such as learning rate,
   local epochs, number of clients, and FedProx coefficient.

---

## Key Finding

The experiments demonstrate that federated learning performance is
strongly influenced by client data distribution. FedAvg achieved
94.28% accuracy under IID partitioning but only 76.54% under the
class-skewed non-IID configuration.

FedProx improved the initial performance under non-IID data but did
not surpass FedAvg after five rounds under the selected configuration.

The experiments also show that communication increases linearly
with the number of communication rounds when full client
participation and fixed-size model transmission are assumed.

Experimental Setup
Parameter	Value
Dataset	MNIST
Model	Simple CNN
Number of clients	5
Communication rounds	5
Local epochs	1
Optimizer	SGD
Learning rate	0.01
FedProx coefficient	0.01
Client participation	Full
Experiments
1. Centralized Baseline

A CNN is trained centrally using the complete training dataset.

This provides a reference point for the federated experiments.

2. FedAvg with IID Data

The training data is randomly distributed among five clients.

Results:

Round	Accuracy
1	84.71%
2	88.94%
3	92.52%
4	93.75%
5	94.28%
3. FedAvg with Non-IID Data

A class-skewed partition is used where each client receives data
primarily from two digit classes.

Results:

Round	Accuracy
1	32.35%
2	62.57%
3	72.17%
4	75.19%
5	76.54%

The final accuracy is 17.74 percentage points lower than the IID
FedAvg experiment under the same general training configuration.

4. FedProx with Non-IID Data

FedProx is evaluated using the same non-IID client distribution.

Results:

Round	Accuracy
1	43.13%
2	62.65%
3	67.80%
4	69.84%
5	72.13%

FedProx achieved higher initial accuracy than FedAvg, but FedAvg
achieved higher accuracy in later rounds under this configuration.

Communication Analysis

The model size and model transmissions were used to estimate
communication cost.

Both server-to-client and client-to-server model transmission are
included.

Round	Cumulative Communication
1	16.08 MB
2	32.17 MB
3	48.25 MB
4	64.34 MB
5	80.42 MB

Communication increases approximately linearly with the number of
communication rounds under full client participation.

Key Findings
FedAvg performed strongly under IID data and reached 94.28%
accuracy after five rounds.
Class-skewed non-IID data significantly reduced performance,
with FedAvg reaching 76.54% after five rounds.
FedProx improved the initial performance under non-IID data but
did not outperform FedAvg after five rounds in this experiment.
Communication cost increased linearly with the number of rounds
because the same model was transmitted between the server and all
participating clients.
Limitations
The experiments use MNIST rather than a large-scale real-world
federated dataset.
Only five clients and five communication rounds were evaluated.
Full client participation is assumed.
The non-IID distribution uses controlled class skew.
Network latency and bandwidth variation are not modeled.
Differential privacy is not experimentally evaluated.
Results depend on model architecture and hyperparameters.
Future Work

Possible extensions include:

Differential privacy experiments.
More realistic heterogeneous client distributions.
Partial client participation.
Larger datasets.
Communication compression.
Quantization and sparsification.
Adaptive local training.
Evaluation across multiple random seeds.
More communication rounds.
Technologies
Python
PyTorch
Torchvision
NumPy
Matplotlib
MNIST
Research Focus

This project focuses on:

Federated Learning
Distributed Machine Learning
Non-IID Data
FedAvg
FedProx
Communication Efficiency
Privacy-Aware Machine Learning

---
