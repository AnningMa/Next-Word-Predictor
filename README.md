# Text Predictor: Next-Word Generation
## Overview
This project implements a real-time next-word prediction engine designed to simulate mobile keyboard suggestions. It compares three distinct architectural approaches—Statistical N-Gram, Feed-Forward Neural Network , and LSTM—to balance prediction accuracy with inference latency.

## Model Evolution & Comparison
We implemented and evaluated three models to demonstrate the evolution from statistical baselines to deep learning sequences.

1. Statistical N-Gram (Baseline)
Pros: Extremely fast inference; simple implementation.

Cons: High sparsity (cannot predict unseen N-grams); lacks long-term context.

2. NGram Neural Network with Embeddings
Improvement: Solved the sparsity problem by introducing Word Embeddings, mapping similar words to nearby vector spaces.

Limitation: Fixed window size (Context Size = [N]) limits the ability to capture variable-length sentence structures.

3. LSTM (Long Short-Term Memory)

Result: Significantly reduced Perplexity (PPL) and improved Top-5 Accuracy.

Performance Benchmark

| **Model** | **Top-5 Accuracy** |
| --------- | ------------------ |
| N-Gram    | `28.70%`           |
| NGram-NN  | `35.66%`           |
| LSTM      | **`42.83%`**       |


## Engineering Trade-offs
While Transformers achieve SOTA results, their $O(N^2)$ complexity creates high latency on edge devices. For a keyboard application, user experience requires response times under [X]ms. LSTM provides the best trade-off between capturing context and running efficiently on CPU.


## Adaptive Personalization
Standard models fail to capture specific user habits. We solved this using Dynamic Interpolation.

Mechanism: P(final) = (1 - α) * P(LSTM) + α * P(UserHistory)

Effect: The model adapts instantly to user vocabulary without expensive fine-tuning.

## Data Processing & Robustness
Data cleaning accounts for 70% of this project's success. The model was trained on the **88milSMS Corpus**, a rigorous academic dataset of authentic French text messages.

* **Source:** The dataset originates from the work of Panckhurst et al. (2014), containing real-world SMS communications, which allows our model to learn authentic casual French (slang, abbreviations).
* **OOV Handling:** Out-of-Vocabulary words are mapped to a special `<UNK>` token to prevent crashes.
* **Sanitization:** Removed non-standard characters, emojis, and chaotic formatting typical in raw SMS data.

## Installation & Usage

Due to GitHub's file size limits, the raw dataset and the pre-trained **N-Gram Neural Network model** are hosted externally.
    
**NGram Neural Network Model**: **https://drive.google.com/file/d/16VKFQj2tx8oRG_S2PaGGRkZu-p1Pjy3_/view?usp=drive_link**

**Raw Data**: http://88milsms.huma-num.fr/corpus.html

---
##  References
1.  **Panckhurst R., Détrie C., Lopez C., Moïse C., Roche M., Verine B. (2014).** *"88milSMS. A corpus of authentic text messages in French"*. Produced by the University Paul-Valéry Montpellier 3 and the CNRS, in collaboration with the Catholic University of Louvain.
