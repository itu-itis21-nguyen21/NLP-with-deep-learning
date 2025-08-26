# Project 3: Neural Machine Translation with Attention

This project implements a Sequence-to-Sequence (Seq2Seq) model with attention for neural machine translation (NMT). This document provides a brief overview of the role each file plays in the project.

---

*   **`run.py`**: The main entry point for the project, orchestrating the training, evaluation, and decoding (inference) of the NMT model.

*   **`nmt_model.py`**: Defines the core Seq2Seq NMT model architecture, including the encoder, decoder, attention mechanism, and the beam search algorithm for translation.

*   **`model_embeddings.py`** Contains the `ModelEmbeddings` class, which creates and manages the learnable embedding layers for both the source and target languages.

*   **`vocab.py`**: Manages vocabulary creation and mapping, using SentencePiece to build vocabularies and providing classes to convert sentences between text and tensors.
