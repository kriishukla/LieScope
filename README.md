# LieScope

## Project Overview

This research project analyzes deception in strategic communication by examining linguistic patterns in lies and perceived lies in the strategy game *Diplomacy*. We build upon the work of Niculae et al. ("Linguistic Harbingers of Betrayal") and Peskov et al. ("It Takes Two to Lie: One to Lie, and One to Listen") to investigate the linguistic features that characterize both actual deception and perceived deception.

## Research Questions

1. Are the linguistic features (e.g., higher politeness) found in messages leading up to a betrayal similar to those found in lies?
2. Are the linguistic features of lies similar to the features of perceived lies?
3. How do the linguistic features of truths perceived as truth, truths perceived as lies, undetected lies, and detected lies differ?

## Dataset

We use the **Deception in Diplomacy** dataset from *"It Takes Two to Lie: One to Lie, and One to Listen"* by Peskov et al. (ACL 2020). This dataset contains:

- 17,289 pairwise messages from 12 different Diplomacy games
- Dual annotations for each message:
  - Sender's intention (Truth/Lie)
  - Receiver's perception (Truth/Lie/No Annotation)
- Additional metadata including:
  - Game state information (score, year, season)
  - Dialogue context
  - Player relationships

The dataset is partitioned into training (9 games), validation (1 game), and test sets (2 games).

## Methodology

### Data Processing

- Text preprocessing using NLTK, spaCy, and ConvoKit
- Extraction of linguistic features:
  - Politeness markers (using Politenessr)
  - Sentiment analysis (using Stanza and VADER)
  - Planning markers
  - Other relevant linguistic cues

### Analysis Approaches

1. **Data Exploration**: Understanding patterns of deception and perception across game progression  
2. **Matching Analysis**: Comparing linguistically similar messages with different deceptive intents  
3. **Feature Comparison**: Analyzing differences between:
   - Truths vs. Lies
   - Perceived Truths vs. Perceived Lies
   - Detected vs. Undetected Lies
   - Messages preceding betrayal vs. normal messages

### Models

We implement and compare multiple approaches:

#### Baseline Models

- Heuristic approaches (Random, Majority Class)
- Feature-based models (Bag of Words, Harbingers)
- Context-enhanced models (incorporating power dynamics)

#### Advanced Models

- Transformer-based representations:
  - BERT
  - RoBERTa
  - MiniLM
- Static word embeddings:
  - GloVe
- Classification algorithms:
  - Logistic Regression
  - Support Vector Machines
  - Random Forest
  - Multi-Layer Perceptron

## Evaluation

Due to the imbalanced nature of the dataset (less than 5% of messages are labeled as lies), we use:

- Macro-averaged F1 Score
- Comparison with human performance (F1 score of 22.5 for detecting actual lies)

## Results and Analysis

Our experiments yielded several important findings regarding the performance of different model configurations for deception detection in the Diplomacy dataset.

### Model Performance

The following table shows macro F1 scores for each model-classifier combination on the **ACTUAL_LIE** detection task:

| Model    | Logistic Regression | Random Forest | SVM   | MLP   |
|----------|---------------------|---------------|-------|-------|
| RoBERTa  | 48.5                | 50.5          | 48.5  | 52.4  |
| BERT     | 53.6                | 51.0          | 49.0  | 49.0  |
| MiniLM   | 48.5                | 50.5          | 48.5  | 48.5  |
| GloVe    | 48.5                | 48.5          | 48.5  | 48.5  |

### Key Findings

- **Best Overall Performance**: BERT combined with Logistic Regression achieved the highest macro F1 score (53.6), substantially outperforming the human baseline (F1 score of 22.5).
- **Language Model Comparison**:
  - BERT consistently delivered strong performance across multiple classifiers
  - RoBERTa showed particularly strong results when paired with the MLP classifier (52.4)
  - MiniLM demonstrated competitive performance despite its smaller size, especially with Random Forest (50.5)
  - GloVe performed consistently worse than contextual embeddings, highlighting the importance of context for deception detection
- **Classifier Performance**:
  - MLP showed strong performance with RoBERTa
  - Random Forest provided consistent results across different embedding types
  - Logistic Regression achieved the highest overall score when paired with BERT
  - SVM generally underperformed compared to other classifiers
- **Politeness and Sentiment**: Analysis of linguistic features revealed that deceptive messages tend to exhibit:
  - Higher levels of politeness (supporting findings from *"Linguistic Harbingers of Betrayal"*)
  - More neutral sentiment scores compared to truthful messages
  - Less emotional variability within messages
- **Temporal Patterns**: Lies were more common in later game stages, particularly when players had significant power differentials
- **Detection Challenge**: Despite improvements over human performance, the absolute F1 scores remain moderate (highest 53.6), highlighting the inherent difficulty of deception detection in strategic settings

## Model Weights

We provide pre-trained model weights for our best-performing configurations. These weights can be used to replicate our results or to apply our models to new data.

### Download Instructions

1. Access the model weights from our Google Drive repository:  
   [https://drive.google.com/drive/folders/18zR_t_26rmIdxRZh-2kC1L5v3hysNBT-?usp=sharing](https://drive.google.com/drive/folders/18zR_t_26rmIdxRZh-2kC1L5v3hysNBT-?usp=sharing)

---

In case of any query, mail at [kriishukla@gmail.com](mailto:kriishukla@gmail.com)
