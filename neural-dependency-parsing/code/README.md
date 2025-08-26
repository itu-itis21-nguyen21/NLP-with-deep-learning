# Project 2: Neural Dependency Parsing

This project builds a transition-based neural dependency parser to analyze the grammatical structure of sentences. This document provides a brief overview of the role each file plays in the project.

---

*   **`run.py`**: The main script to orchestrate the entire process. It handles data loading, initializes the model, runs the training loop, saves the best model weights, and performs the final evaluation.

*   **`parser_model.py`**: Defines the feed-forward neural network in PyTorch. This file contains the model's architecture, including the embedding lookup, hidden layers, and the forward pass that predicts parsing transitions.

*   **`parser_transitions.py`**: Implements the core logic of the transition-based parser. It defines the `PartialParse` state (stack, buffer, dependencies) and contains the `minibatch_parse` algorithm that drives the parsing process for batches of sentences.