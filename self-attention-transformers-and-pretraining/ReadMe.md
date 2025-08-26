# Project 4: Self-Attention, Transformers, and Pretraining

This project implements a GPT-style Transformer model from scratch, explores the paradigm of pretraining and fine-tuning, and introduces a more efficient attention variant inspired by the Perceiver architecture. This document provides a brief overview of the role each file in the **src** folder plays in the project.

---

### Core Logic & Orchestration

*   **`run.py`**: The main entry point for the project. This script handles command-line arguments to orchestrate the entire workflow, including pretraining, fine-tuning, and evaluation of the different model variants.

*   **`model.py`**: Defines the GPT model architecture. It contains the implementation for the standard Transformer `Block` as well as the `DownProjectBlock` and `UpProjectBlock` for the more efficient "Perceiver" variant. This file assembles the full model from its constituent parts.

*   **`attention.py`**: Implements the core attention mechanisms. It contains the `CausalSelfAttention` module for standard GPT blocks and the `CausalCrossAttention` module used by the Perceiver variant.

### Training & Data Handling

*   **`trainer.py`**: Contains a generic `Trainer` class that handles the model training and evaluation loops. It implements the optimization logic (AdamW), learning rate scheduling, and checkpoint saving.

*   **`dataset.py`**: Manages all data loading and preprocessing.
    *   `CharCorruptionDataset`: Implements the span corruption objective for the pretraining task.
    *   `NameDataset`: Formats the name-birthplace data for the fine-tuning and evaluation tasks.

### Utilities & Evaluation

*   **`utils.py`**: A collection of helper functions used across the project. This includes functions for setting random seeds, sampling text from the model (`sample`), and evaluating the model's accuracy on the birthplace prediction task (`evaluate_places`).

*   **`london_baseline.py`**: A simple script to compute a non-ML baseline for the fine-tuning task. It calculates the accuracy that would be achieved by always predicting "London" as the birthplace.