# Deep Learning for NLP Projects

This repository contains a collection of projects completed as part of my self-study in Deep Learning for Natural Language Processing. These projects cover a wide range of fundamental and advanced topics, starting from basic neural networks and word embeddings, and progressing to modern Transformer architectures like BERT. Each project involves both theoretical understanding (through gradient derivations) and practical implementation in Python using libraries like NumPy and PyTorch.

## Core Concepts & Technologies

Across these projects, the following key concepts and technologies were explored and implemented:

*   **Programming:** Python
*   **Libraries:** PyTorch, NumPy
*   **Neural Network Fundamentals:**
    *   Activation Functions (Sigmoid, Softmax)
    *   Forward & Backward Propagation
    *   Gradient Derivations & Gradient Checking
*   **Word Embeddings:**
    *   Word2Vec (Skip-gram, CBOW)
    *   Negative Sampling Loss
*   **Sequence Models:**
    *   Recurrent Neural Networks (RNNs)
    *   LSTMs (Bidirectional & Unidirectional)
*   **Advanced Architectures:**
    *   Sequence-to-Sequence (Seq2Seq) Models
    *   Attention Mechanisms (Dot-Product, Multiplicative)
    *   Self-Attention & Multi-Head Self-Attention
    *   Transformers
    *   BERT (minBERT implementation)
*   **Training & Optimization:**
    *   Stochastic Gradient Descent (SGD)
    *   Adam Optimizer
    *   Regularization (Dropout)
*   **NLP Tasks:**
    *   Sentiment Analysis
    *   Dependency Parsing
    *   Neural Machine Translation (NMT)
    *   Paraphrase Detection
    *   Semantic Textual Similarity (STS)
*   **Training Paradigms:**
    *   Pretraining & Fine-tuning
    *   Multi-task Learning

---

## Project Portfolio

### Project 1: Neural Network Fundamentals & Word2Vec

This project focuses on the foundational concepts of neural networks and word embeddings. It builds the core components necessary for more complex models, starting from the mathematical principles.

*   **Key Implementations:**
    *   Derived gradients for `sigmoid` and `softmax` activation functions.
    *   Implemented a simple one-hidden-layer neural network with forward and backward propagation.
    *   Implemented `word2vec` models (Skip-gram and CBOW) to learn word embeddings.
    *   Implemented both Softmax and Negative Sampling loss functions for training word vectors.
    *   Trained custom word embeddings from scratch using Stochastic Gradient Descent (SGD).
    *   Applied the trained word vectors to a downstream Sentiment Analysis task.

### Project 2: Neural Dependency Parsing

This project involved building a transition-based neural dependency parser. The goal was to analyze the grammatical structure of sentences by predicting a sequence of transitions (SHIFT, LEFT-ARC, RIGHT-ARC) to form a dependency tree.

*   **Key Implementations:**
    *   Understood and implemented advanced optimization (Adam Optimizer) and regularization (Dropout) techniques.
    *   Implemented the mechanics of a transition-based dependency parsing system (stack, buffer, transitions).
    *   Built a neural network classifier in PyTorch to predict the correct parsing transition at each step.
    *   Trained the parser on a treebank and evaluated its performance using the Unlabeled Attachment Score (UAS).
    *   Performed an error analysis to understand the model's shortcomings.

### Project 3: Neural Machine Translation with Attention

This project focused on building a Neural Machine Translation (NMT) system using a Sequence-to-Sequence (Seq2Seq) architecture with an attention mechanism.

*   **Key Implementations:**
    *   Implemented a full Seq2Seq model for translation.
    *   Used a Bidirectional LSTM for the encoder and a Unidirectional LSTM for the decoder.
    *   Implemented a multiplicative attention mechanism from scratch to allow the model to focus on relevant parts of the source sentence during translation.
    *   Trained the NMT system on a parallel corpus.
    *   Evaluated the model's translation quality using the BLEU score and analyzed its outputs.

### Project 4: Self-Attention, Transformers, and Pretraining

This project delved into the core components of the Transformer architecture and explored the modern paradigm of pretraining and fine-tuning.

*   **Key Implementations:**
    *   Explored the mathematical properties of the self-attention mechanism, understanding how it can copy and average information.
    *   Implemented a Transformer model for a language modeling task.
    *   **Pretrained** a Transformer on a large text corpus using a span corruption objective.
    *   **Fine-tuned** the pretrained model for a downstream, knowledge-intensive question-answering task.
    *   Researched and implemented a more efficient attention variant (Perceiver-style cross-attention) to handle long contexts.

### Project 5: Implementing and Fine-Tuning minBERT for Downstream Tasks

This capstone project involved implementing a minimalist version of BERT ("minBERT") and then fine-tuning it to create robust sentence embeddings for a variety of NLP tasks.

*   **Key Implementations:**
    *   Implemented the core components of the BERT architecture, including Multi-Head Self-Attention and Transformer encoder layers.
    *   Loaded pretrained weights into the custom BERT implementation to leverage its learned knowledge.
    *   Fine-tuned the model on a single downstream task: Sentiment Analysis.
    *   Implemented a **multi-task learning** framework to fine-tune the model simultaneously on three different tasks: Sentiment Analysis, Paraphrase Detection, and Semantic Textual Similarity (STS).