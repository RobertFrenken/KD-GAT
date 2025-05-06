# KD-GAT: Combining Knowledge Distillation and Graph Attention Transformer for a Controller Area Network Intrusion Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

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

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- Other dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CAN-Graph.git
   cd CAN-Graph


## Datasets
This repository provides several datasets for training, testing, and analysis. See the table below for details and download links.

| Name      | Description                  | Link                                      |
|-----------|-----------------------------|----------------------------------------------------|
| Car Survival | Sample training data        | [Link](https://ocslab.hksecurity.net/Datasets/survival-ids)      |
| CAR-Hacking Dataset| Test data for evaluation    | [Link](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)     |
| can-train-and-test | Additional unlabeled data   | [Link](https://bitbucket.org/brooke-lampe/can-train-and-test-v1.5/src/master/)  |
