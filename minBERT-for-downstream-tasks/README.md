# Project 5: minBERT for Downstream Tasks

This project implements a minimalist version of BERT ("minBERT") and fine-tunes it on various NLP downstream tasks like sentiment analysis, paraphrase detection, and semantic textual similarity.

---

*   **`config.py`**: Defines `PretrainedConfig` and `BertConfig` classes, which are used to store model configurations and load pretrained configurations.

*   **`tokenizer.py`**: Implements the `BertTokenizer` for tokenizing text, converting tokens to IDs, and handling special tokens, essential for preparing text input for BERT.

*   **`base_bert.py`**: Provides the base `BertPreTrainedModel` class, handling common functionalities like weight initialization and loading pretrained weights.

*   **`bert.py`**: Implements the core BERT model architecture, including `BertSelfAttention` (multi-head self-attention), `BertLayer` (Transformer encoder block), and the overall `BertModel` structure.

*   **`datasets.py`**: Manages data loading and preprocessing for all three downstream tasks, including the creation of custom PyTorch `Dataset` classes and collate functions.

*   **`optimizer.py`**: Implements the AdamW optimizer, used for training of the BERT-based models.

*   **`classifier.py`**: Implements the `BertSentimentClassifier` specifically for sentiment analysis on the SST dataset and handles its training and evaluation.

*   **`multitask_classifier.py`**: Defines the `MultitaskBERT` model, which extends BERT to handle sentiment classification, paraphrase detection, and semantic textual similarity tasks simultaneously. It also orchestrates the multi-task training loop.

*   **`evaluation.py`**: Contains functions to evaluate the performance of the models on individual tasks (SST, Quora, STS) and provides a combined evaluation for the multitask model.



