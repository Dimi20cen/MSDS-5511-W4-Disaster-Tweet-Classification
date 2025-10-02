# Disaster Tweet Classification

A notebook project that classifies tweets as **real disaster** or **not disaster**. The dataset comes from Kaggle’s “NLP Getting Started” competition (7.6k training tweets, 3.2k test tweets).

## Results (Validation Set)

* **Bidirectional LSTM model**

  * Accuracy: 79.4%
  * Precision: 0.76
  * Recall: 0.77
  * F1: 0.76

## Approach

1. **EDA & Cleaning:**

   * Lowercasing, remove URLs/HTML/punctuation.
   * Stopword removal & lemmatization (NLTK).
   * Fill missing keywords with `nokeyword`; prepend keyword to tweet text.
   * Dropped messy location feature.
2. **Model:**

   * Tokenization (15k vocab), padded to length 60.
   * **Embedding → Bidirectional LSTM → Dense layers** with Dropout.
   * Output: sigmoid for binary classification.
3. **Tuning & Training:**

   * Keras Tuner RandomSearch for embedding dim, LSTM units, dropout, dense size, learning rate.
   * Early stopping + best weights checkpointing.

