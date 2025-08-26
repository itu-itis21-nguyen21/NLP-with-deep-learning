# Project 1: Neural Network Fundamentals & Word2Vec

This project covers the foundational concepts of neural networks and word embeddings, implementing key components from scratch using NumPy.

---

*   **`q1_softmax.py`**: Implements the softmax activation function, optimized for numerical stability and performance.

*   **`q2_sigmoid.py`**: Implements the sigmoid activation function and its gradient.

*   **`q2_neural.py`**: Implements a two-layer sigmoidal neural network. It includes the `forward_backward_prop` function for computing both the forward pass (cost) and the backward pass (gradients) for all network parameters.

*   **`q3_word2vec.py`**: Contains the core implementation of Word2Vec model.

*   **`q3_sgd.py`**: Implements the Stochastic Gradient Descent (SGD) optimization algorithm, including learning rate annealing and parameter saving.

*   **`q3_run.py`**: The main script to train word vectors using the implemented Skip-gram model with negative sampling. It loads the Stanford Sentiment Treebank dataset, trains word embeddings, and visualizes a subset of them.

*   **`q4_sentiment.py`**: Applies the trained word vectors to a sentiment analysis task using logistic regression. It includes functions for featurizing sentences, tuning regularization, and evaluating model performance with accuracy and confusion matrices.