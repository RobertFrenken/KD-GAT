# KD-GAT: Combining Knowledge Distillation and Graph Attention Transformer for a Controller Area Network Intrusion Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
<!-- ![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg) -->
---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [License](#license)
- [TODO](#TODO)
---
## Description

KD-GAT is a machine learning framework that combines **Knowledge Distillation (KD)** and **Graph Attention Networks (GAT)** to create a robust intrusion detection system for Controller Area Networks (CAN). This project aims to improve cybersecurity in automotive systems by detecting malicious activities on the CAN bus.

The framework leverages:
- **Knowledge Distillation**: To train lightweight student models for deployment on edge devices.
- **Graph Attention Networks**: To model relationships between CAN messages effectively.

---

## Features

- **Knowledge Distillation**: Train teacher-student models for efficient inference.
- **Graph-Based Learning**: Use GATs to process CAN data as graph representations.
- **Support for Multiple Datasets**: Preprocessing pipelines for various CAN datasets.
- **Custom Loss Functions**: Includes Focal Loss to address class imbalance.
- **Reproducibility**: Configurable training and evaluation pipelines with YAML files.

---
### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- Other dependencies listed in `requirements.txt`

## Installation
```bash
    git clone https://github.com/robertfrenken/CAN-Graph.git
    cd CAN-Graph
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
```
## Usage
To train the teacher and student models:
```bash
python osc-training.py
```
To evaluate the models:
```bash
python evaluation.py
```
## Datasets
Datasets should be in a folder called datasets. This pro

| Name      | Description                  | Link                                      |
|-----------|-----------------------------|----------------------------------------------------|
| can-train-and-test | Primary dataset. Contains all 3 datasets.   | [Link](https://bitbucket.org/brooke-lampe/can-train-and-test-v1.5/src/master/)  |
| Car Survival | For reference. Use format from above.     | [Link](https://ocslab.hksecurity.net/Datasets/survival-ids)      |
| CAR-Hacking Dataset| For reference. Use format from above.   | [Link](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)     |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## TODO

- [ ] Add details about base.yaml file
- [ ] Improve preprocessing pipelines for additional datasets.
- [ ] Add support for real-time CAN data processing.
- [ ] Publish pre-trained models for public use.
