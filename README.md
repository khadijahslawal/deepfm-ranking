# DeepFM for Search Relevance Ranking on MS MARCO

> Applying Deep Factorization Machines to passage ranking using the MS MARCO dataset

> University of Chicago — M.S in Applied Data Science

> Contributors: Amulya Rayasam, Grace Rowan, Khadija Shuaib

---

## Table of Contents
1. [Problem Introduction & Dataset](#1-problem-introduction--dataset)
2. [Model: DeepFM](#2-model-deepfm)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture & Training](#4-model-architecture--training)
5. [Feature Importance Analysis](#5-feature-importance-analysis)
6. [Model Results](#6-model-results)
7. [Reproducibility](#7-reproducibility)
8. [Repository Structure](#8-repository-structure)
9. [Setup & Installation](#9-setup--installation)

---

## 1. Problem Introduction & Dataset

### The Problem with Search Ranking

Search is everywhere; Google, Amazon, Instagram, your laptop's file system. 
The goal of search is not just to return relevant results, but to return them 
**in the right order**.

Consider this: if the best answer to your query is ranked 10th and an irrelevant 
result is ranked 1st, the search technically "worked" — but the user experience 
is broken. For e-commerce sites and search engines, poor ranking order directly 
translates to lost users and lost revenue.

> **Core problem:** *How do we ensure search results are ranked in the right order 
> of answering the given query?*

<img width="808" height="402" alt="Search Ranking" src="https://github.com/user-attachments/assets/350c51c4-6156-40bb-8c40-7db76579f899" />


### Dataset: MS MARCO v2.1
[MS MARCO](https://microsoft.github.io/msmarco/) (Microsoft Machine Reading Comprehension) is a large-scale information retrieval benchmark built from real Bing search queries.

```python
from datasets import load_dataset
dataset = load_dataset("ms_marco", "v2.1", split="validation")
```
#### Raw Dataset
The dataset is publicly available on Hugging Face. The raw structure contains 
nested passages per query — each query maps to a `passages` dict with 
`passage_text` (list of candidate passages) and `is_selected` (binary labels).

[Explore the dataset on Hugging Face](https://huggingface.co/datasets/microsoft/ms_marco)

| Property | Value |
|----------|-------|
| Split used | Validation |
| Raw queries | 101,093 |
| Sample used | 10,000 queries |
| After explosion | ~99,840 query-passage pairs |
| Positive rate | ~10.7% (one relevant passage per query) |
| Query types | NUMERIC, DESCRIPTION, ENTITY, LOCATION, PERSON |

#### Label Structure

MS MARCO uses **pointwise binary labeling** — each query has typically one relevant passage (`is_selected=1`) and several irrelevant ones (`is_selected=0`). DeepFM learns to score the relevant passage higher than irrelevant ones, enabling ranking at inference time.

#### Data Pipeline

```
Raw dataset (10k rows, nested passages)
        ↓
Explode passages → each row = one query-passage pair (~99,840 rows)
        ↓
Sample 10,000 queries → explode → 99,840 query-passage pairs
        ↓
Filter queries with no relevant passage
        ↓
Feature engineering
        ↓
Train/Val/Test split by query_id (70/15/15)
```

> **Important**: We split by `query_id` (not by row) to ensure all passages for a given query stay in the same split. Splitting by row would break query groups and produce artificially inflated ranking metrics.

### Task
Given a search query and a set of candidate passages, rank the passages by relevance so that the most relevant passage appears at the top. This is the **passage re-ranking** problem; a core component of modern search engines.

---

## 2. Model: DeepFM

### Overview
DeepFM (Deep Factorization Machine) is a neural architecture that jointly trains:
- A **Factorization Machine (FM)** component — learns low-order feature interactions
- A **Deep Neural Network (DNN)** component — learns high-order non-linear interactions

over shared input embeddings. Introduced by [Guo et al. (2017)](https://arxiv.org/abs/1703.04247), DeepFM learns both low-order and high-order feature interactions end-to-end without manual feature crossing.

### What DeepFM does? 
| Property | Benefit |
|----------|---------|
| FM component | Automatically learns pairwise feature interactions (e.g. high BM25 + short passage) |
| Deep component | Captures complex patterns beyond pairwise (e.g. query type × passage features) |
| Shared embeddings | FM and DNN learn from the same representation — no information loss |
| Binary task | Directly optimizable for binary relevance labels |


### Why DeepFM for Search Ranking?

**Factorization Machine Layer**

In search ranking, you have lots of features about a query-document pair: BM25 score, document length, query length, click rate, term overlap, etc. A linear model treats these independently. But interactions between features matter; a high BM25 score means something different depending on query length (short navigational queries behave differently from long informational ones).

Factorization Machines learn these pairwise feature interactions automatically without having to manually engineer them. Each feature gets a learned vector (embedding), and interactions are computed as dot products between those vectors. This is efficient and works well even with sparse data.

**DNN Layer**

DeepFM extends this by stacking a deep neural network on top of the FM layer. The FM component captures low-order interactions (pairs), while the DNN captures higher-order and non-linear ones. Both share the same input embeddings. The final prediction combines both. 

> This is the architecture that powers ranking in Alibaba's ad system, Huawei's app store, and is foundational to how modern search rankers are designed.


<img width="890" height="570" alt="Screenshot 2026-03-10 at 4 18 52 pm" src="https://github.com/user-attachments/assets/e1ef9a06-f195-4491-b1d2-8b40312cc3ef" />

### Training Algorithm

**Input:** Training set $\{(x_i, y_i)\}_{i=1}^N$ for i = 1,...,N, learning rate $\eta$, embedding dim $k$, DNN layers $[h_1, h_2, \ldots]$

**Algorithm:**
1. Initialize embedding vectors $\mathbf{v}_i \sim \mathcal{N}(0, 0.01)$ for all features
2. Initialize DNN weights $\mathbf{W}^{(l)}$ via Xavier initialization
3. **For** each mini-batch $\mathcal{B}$:
   - a. Compute $y_{\text{FM}}$ via first + second order terms
   - b. Compute $y_{\text{DNN}}$ via forward pass through hidden layers
   - c. Compute $\hat{y} = \sigma(y_{\text{FM}} + y_{\text{DNN}})$
   - d. Compute loss $\mathcal{L}$
   - e. Backpropagate gradients through both components (shared embedding receives gradients from both paths)
   - f. Update all parameters via Adam: $\theta \leftarrow \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$
4. Evaluate on validation set after each epoch
5. Early stop if validation loss does not improve for $p$ epochs

**Inference (Ranking):**
1. For a query $q$ and candidate passages $\{p_1, \ldots, p_K\}$
2. Extract features $\mathbf{x}_{q,p_j}$ for each pair
3. Compute $\hat{y} = \sigma(y_{\text{FM}} + y_{\text{DNN}})$
4. Return passages sorted by $\hat{y}$ in descending order

### Ranking Mechanism 
Given candidate passages for each query, there are a couple of options we have with regards to ranking 
1. Point Wise Ranking
2. List Wise Ranking
   
<img width="868" height="469" alt="Screenshot 2026-03-10 at 4 59 03 pm" src="https://github.com/user-attachments/assets/e668cba2-4393-4b0f-8102-8fc4899207e6" />


#### Point Wise Ranking
- The Pointwise Approach: We treat each query-passage pair as an independent data point. The model predicts a "relevance score" (probability from 0 to 1) for every candidate.
- The Ranking: Even though the training data only has one "1," the model outputs a decimal score (e.g., 0.85, 0.42, 0.12) for all candidates. We then sort the list by these scores.
- The Goal: If the model is working, that single "selected" if is_selected = 1 passage should end up with the highest score and sit at Rank #1.

#### Listwise Ranking (Search Engine Standard)
- The Concept: The model looks at the entire list of candidate documents for a query all at once and tries to optimize the order to maximize a metric like NDCG.
- The Challenge: Implementation is much harder. Many libraries don't support listwise loss functions natively without complex custom code.

> We used a point wise ranking

### Package Selection

Several Python libraries support factorization machine based models for ranking and CTR prediction:

| Library | Language | Notes |
|---------|----------|-------|
| [`deepctr-torch`](https://github.com/shenweichen/DeepCTR-Torch) | PyTorch | DeepFM + many CTR models, GPU support, active maintenance |
| [`deepctr`](https://github.com/shenweichen/DeepCTR) | TensorFlow/Keras | Same author, TF version of above |
| [`xlearn`](https://github.com/aksnzhy/xlearn) | C++/Python | Fast FM and FFM, but no deep component |
| [`pytorch-fm`](https://github.com/rixwew/pytorch-factorization-machines) | PyTorch | Lightweight FM implementations, limited model variety |
| [`RecBole`](https://recbole.io/) | PyTorch | Full recommendation framework, includes DeepFM but heavy overhead |

**We selected `deepctr-torch` for the following reasons:**

- **Direct DeepFM implementation**: purpose-built for DeepFM and other CTR models with a clean, well-documented API
- **PyTorch backend**: native GPU support, compatible with our training environment
- **`SparseFeat` and `DenseFeat` abstractions**: handles our mixed feature types (dense + sparse) cleanly without manual preprocessing
- **Lightweight**: unlike RecBole, it imposes no framework overhead or rigid data format requirements, making it straightforward to integrate into a custom pipeline
- **Active maintenance**: well-maintained with clear documentation and examples

```bash
pip install deepctr-torch
```

```python
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
```

### Mathematical Formulation

**Input Representation**

$$\mathbf{x} \in \mathbb{R}^d$$

where $d$ = total number of features. Features are either dense (continuous) or sparse (categorical). Each sparse feature field $i$ has an embedding vector $\mathbf{v}_i \in \mathbb{R}^k$ where $k$ is the embedding dimension.

**FM Component — First-Order Term**

$$y_{\text{order1}} = w_0 + \sum_{i=1}^{d} w_i x_i$$

**FM Component — Second-Order Interaction Term**

$$y_{\text{order2}} = \frac{1}{2} \sum_{l=1}^{k} \left[ \left(\sum_{i=1}^{d} v_{il} x_i \right)^2 - \sum_{i=1}^{d} v_{il}^2 x_i^2 \right]$$

This reduces computation from $O(d^2)$ to $O(kd)$.

**DNN Component**

$$\mathbf{a}^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} \mathbf{a}^{(l)} + \mathbf{b}^{(l)} \right)$$

**Final Prediction**

$$\hat{y} = \sigma \left( y_{\text{FM}} + y_{\text{DNN}} \right)$$

where $\sigma$ is sigmoid for binary relevance prediction.

---

## 3. Feature Engineering

The raw MS MARCO dataset provides only: `query`, `query_id`, `query_type`,  and nested `passages` (containing `passage_text` and `is_selected`). **All features were engineered from scratch.** 

#### Baseline Features 

We began with 9 baseline features covering the core signals for passage ranking:

| Feature | How it's computed |
|---------|------------------|
| `query_length` | Word count of query string |
| `passage_length` | Word count of passage string |
| `length_ratio` | query_length / (passage_length + 1) |
| `exact_match` | Binary — full query string appears verbatim in passage |
| `query_term_coverage` | Fraction of query words found in passage |
| `jaccard_similarity` | Jaccard similarity between query and passage word sets |
| `tfidf_cosine_sim` | TF-IDF cosine similarity (n-grams 1-2, 50k vocab) |
| `bm25_score` | BM25 relevance score computed via `rank_bm25` |
| `passage_position` | Index of passage in the original candidate list |

#### Enriched Features
We further expanded to **22 features across 4 categories**,  adding richer lexical, numeric, and query-type signals.
All features are **dense**  (continuous/binary) except `query_type_encoded` which is sparse (categorical embedding).


### Features Category 1: Lexical Matching

*"Do the query words actually appear in the passage? Do they mean the same thing, even with different words?"*

| Feature | Description |
|---------|-------------|
| `bm25_score` | BM25 relevance score between query and passage |
| `tfidf_cosine_sim` | TF-IDF cosine similarity (n-grams 1-2, 50k vocab) |
| `query_term_coverage` | Fraction of query words found in passage |
| `exact_match` | Binary — does the full query appear verbatim in passage |
| `idf_weighted_coverage` | Query term coverage weighted by IDF (rare words count more) |
| `bigram_overlap` | Fraction of query bigrams found in passage |
| `trigram_overlap` | Fraction of query trigrams found in passage |
| `jaccard_similarity` | Jaccard similarity between query and passage word sets |
| `max_tfidf_term` | Maximum TF-IDF score of any query term in the passage |

### Features Category 2: Statistical / Length Features

*"What does the structure of the passage tell us?" e.g  Longer passages may dilute relevance*

| Feature | Description |
|---------|-------------|
| `query_length` | Number of words in query |
| `passage_length` | Number of words in passage |
| `length_ratio` | query_length / (passage_length + 1) |
| `passage_position` | Index of passage in the original candidate list |

### Features Category 3: Numeric Content Features

*Does the query ask for a number, and does the passage deliver one? "how many calories in an egg?" should contain digits.*

| Feature | Description |
|---------|-------------|
| `passage_has_number` | Binary — passage contains digits |
| `query_has_number` | Binary — query contains digits |
| `both_have_number` | Binary — both query and passage contain digits |

### Features Category 4: Query Type Interaction Features

*What kind of answer does this query expect, and does the passage match that intent? e.g a description query rewards longer explanatory passages, an entity query rewards exact name matches*

| Feature | Description |
|---------|-------------|
| `is_numeric` | Binary — query type is NUMERIC |
| `is_description` | Binary — query type is DESCRIPTION |
| `is_entity` | Binary — query type is ENTITY |
| `numeric_x_has_number` | is_numeric × both_have_number |
| `description_x_length` | is_description × passage_length |
| `entity_x_exact_match` | is_entity × exact_match |
| `query_type_encoded` | Label-encoded query type (SparseFeat — gets its own embedding) |

### BM25 Computation Note

BM25 is a widely used, keyword-based ranking function in information retrieval that ranks documents based on query term frequency, inverse document frequency, and document length normalization. It improves upon traditional TF-IDF by reducing the impact of high-frequency words, making it the standard ranking algorithm for search engines like Elasticsearch, Lucene, and OpenSearch

<img width="692" height="122" alt="Screenshot 2026-03-10 at 5 10 55 pm" src="https://github.com/user-attachments/assets/824a9f80-158f-49b2-b156-9292045ae53d" />

<img width="844" height="122" alt="Screenshot 2026-03-10 at 5 11 37 pm" src="https://github.com/user-attachments/assets/cb7b18c0-543c-4aa6-b07e-3fc2032db395" />


We computed it during feature engineering (not pre-extracted). To avoid recomputing for each row, we score each **unique query** once against all passages, then map scores back:

```python
# Efficient: score each unique query once (~6,980 calls vs ~99,840)
unique_queries = df_exploded[['query_id', 'query']].drop_duplicates()
bm25_score_map = {}
for _, row in tqdm(unique_queries.iterrows(), total=len(unique_queries)):
    query_tokens = str(row['query']).lower().split()
    bm25_score_map[row['query_id']] = bm25.get_scores(query_tokens)
```

### Sharing Pre-computed Features
Feature engineering outputs are saved as pickle files. 

```python
df = pd.read_pickle('/content/drive/MyDrive/YOUR_FOLDER/df_features_enriched.pkl')
```

---

## 4. Model Architecture & Training

### Final Model Configuration
```python
model = DeepFM(
    linear_feature_columns=fixlen_feature_columns,
    dnn_feature_columns=fixlen_feature_columns,
    dnn_hidden_units=(128, 64, 32),
    dnn_dropout=0.3,
    task='binary',
    device=device
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_crossentropy', 'auc']
)
```

### Train / Val / Test Split
```python
# Split by query_id — keeps all passages for a query in the same split
unique_qids = df['query_id'].unique()
train_qids, temp_qids = train_test_split(unique_qids, test_size=0.30, random_state=42)
val_qids, test_qids   = train_test_split(temp_qids,   test_size=0.50, random_state=42)
```

| Split | Queries | Rows |
|-------|---------|------|
| Train | ~6,986 | ~69,853 |
| Val | ~1,497 | ~14,966 |
| Test | ~1,497 | ~14,966 |



### Evaluation Metrics 

**1. NDCG@10** (Normalized Discounted Cumulative Gain): Measures whether the most relevant documents are ranked highest, with a discount for lower positions. This is the gold standard for search evaluation and what platforms like Bing optimizes for.

$$\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}, \quad \text{DCG@}K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

```python
def compute_ndcg_at_k(df_eval, preds, k=10):
    df_eval = df_eval.copy()
    df_eval["pred"] = preds
    ndcg_scores = []

    for query_id, group in df_eval.groupby("query_id"):
        group_sorted = group.sort_values("pred", ascending=False)
        relevance = group_sorted["label"].values[:k]

        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance))

        ideal = group.sort_values("label", ascending=False)["label"].values[:k]
        idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
```

**2. MRR** (Mean Reciprocal Rank): What rank did the first relevant document appear at?

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the position of the first relevant passage for query $q$.

```python
def compute_mrr(df_eval, preds):
    df_eval = df_eval.copy()
    df_eval["pred"] = preds
    reciprocal_ranks = []

    for query_id, group in df_eval.groupby("query_id"):
        group_sorted = group.sort_values("pred", ascending=False)
        relevance = group_sorted["label"].values

        rr = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
```

**3. MAP** (Mean Average Precision): Across all queries, how consistently is the model ranking relevant passages near the top

$$\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}(q), \quad \text{AP}(q) = \frac{1}{R_q} \sum_{k=1}^{K} P(k) \cdot \text{rel}(k)$$

where $R_q$ is the total number of relevant passages for query $q$, $P(k)$ is precision at rank $k$, and $\text{rel}(k) = 1$ if the passage at rank $k$ is relevant.


```python
def compute_map(df_eval, preds):
    df_eval = df_eval.copy()
    df_eval["pred"] = preds
    ap_scores = []

    for query_id, group in df_eval.groupby("query_id"):
        group_sorted = group.sort_values("pred", ascending=False)
        relevance = group_sorted["label"].values

        if relevance.sum() == 0:
            continue

        num_relevant = 0
        precisions = []

        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                num_relevant += 1
                precisions.append(num_relevant / rank)

        ap_scores.append(np.mean(precisions))

    return float(np.mean(ap_scores)) if ap_scores else 0.0
```
**4. AUC** (Area Under the ROC Curve):

$$\text{AUC} = P(\hat{y}_{\text{relevant}} > \hat{y}_{\text{irrelevant}})$$

where $\hat{y}$ is the model's predicted relevance score. AUC measures the probability 
that a randomly chosen relevant passage is scored higher than a randomly chosen 
irrelevant one.

### Sensitivity Analysis
We ran hyperparameter sensitivity analysis varying architecture size and dropout, using early stopping (`patience=3`) to ensure fair comparison:

| Architecture | Dropout | NDCG@10 | MAP |
|-------------|---------|---------|-----|
| (128, 64, 32) | 0.3 | 0.6139 | 0.4899 |
| (128, 64, 32) | 0.5 | 0.6159 | 0.4926 |
| (256, 128, 64) | 0.3 | 0.6142 | 0.4901 |
| (128, 64, 32) | 0.0 | 0.6129 | 0.4884 |
| (128, 64, 32) | 0.1 | 0.6113 | 0.4863 |

**Key findings:**
- All four (128, 64, 32) configurations produced NDCG@10 within 0.005 of each other — model is robust to dropout in the 0.0–0.5 range
- Dropout=0.0 showed the weakest performance, suggesting some regularization is beneficial
- The larger (256, 128, 64) architecture was competitive but did not meaningfully outperform the simpler (128, 64, 32) architecture — unnecessary complexity for 22 tabular features
- We selected **(128, 64, 32) with dropout=0.3** as the final architecture based on performance, simplicity, and standard regularization practice

**Selected Model (128, 64, 32) with dropout=0.3 Training**

<img width="938" height="314" alt="Screenshot 2026-03-10 at 6 26 36 pm" src="https://github.com/user-attachments/assets/d74e28a5-ed42-4723-9c8c-5fb53dcfb6b7" />

---

## 5. Feature Importance Analysis

### Permutation Importance
We measured feature importance by shuffling each feature on the trained model and measuring the resulting performance drop (averaged over 5 repeats):

| Feature | NDCG@10 Drop ± Std | Interpretation |
|---------|-------------------|----------------|
| `tfidf_cosine_sim` | +0.1173 ± 0.0054 | Dominant feature by far — the trained model relies on semantic vocabulary overlap above all else |
| `query_term_coverage` | +0.0216 ± 0.0030 | Strong lexical matching signal |
| `passage_has_number` | +0.0211 ± 0.0040 | Numeric content is highly discriminative for factual queries |
| `idf_weighted_coverage` | +0.0193 ± 0.0020 | IDF-weighted matching outperforms raw coverage |
| `is_numeric` | +0.0183 ± 0.0032 | Query type signal — model actively uses query intent |
| `passage_position` | +0.0135 ± 0.0044 | Structural position carries meaningful relevance signal |
| `both_have_number` | +0.0124 ± 0.0050 | Numeric content interaction consistently relied upon |
| `passage_length` | +0.0119 ± 0.0013 | Passage quality indicator |
| `jaccard_similarity` | +0.0098 ± 0.0009 | Consistent word overlap signal |
| `max_tfidf_term` | +0.0078 ± 0.0050 | Meaningful but variable contribution |
| `is_description` | +0.0071 ± 0.0029 | Query type signal for description queries |
| `bm25_score` | +0.0036 ± 0.0022 | Genuine positive contributor — modest but present |
| `trigram_overlap` | -0.0009 ± 0.0005 | Only noise feature — too sparse beyond TF-IDF signal |

![](images/permutation_importance.png "Permutation Importance")

### Feature Ablation
We retrained the model with one feature dropped at a time (10 epochs, same seed) to measure each feature's contribution during learning:

| Tier | Features | NDCG@10 Drop | Interpretation |
|------|---------|-------------|----------------|
| High | `passage_position` | +0.0111 | Structural position is the single strongest learning signal |
| High | `passage_length` | +0.0070 | Longer passages tend to be more answer-rich and complete |
| High | `passage_has_number`, `entity_x_exact_match`, `idf_weighted_coverage`, `query_term_coverage`, `is_entity` | +0.006–0.007 | Numeric content and query-passage interaction features are surprisingly informative |
| Medium | `is_description`, `is_numeric`, `query_length`, `description_x_length`, `query_has_number`, `length_ratio` | +0.004–0.006 | Query type and structural features provide moderate collective contribution |
| Low | `bm25_score`, `both_have_number`, `max_tfidf_term`, `numeric_x_has_number`, `jaccard_similarity`, `trigram_overlap`, `tfidf_cosine_sim`, `exact_match` | +0.001–0.004 | Weak but genuine positive contributors |
| Noise | `bigram_overlap` | -0.0048 | Only feature that meaningfully hurts performance |

**Key takeaways from both analyses:**
- **`tfidf_cosine_sim` is the dominant feature in permutation importance** (+0.1173) — its drop is more than 5x larger than the next feature, revealing the trained model's heavy reliance on semantic vocabulary overlap
- **`passage_position` leads in ablation** (+0.0111) but ranks 6th in permutation — the model learns from positional ordering but ultimately relies more on semantic similarity at inference time
- **21 out of 22 features contribute positively in ablation** — the feature engineering process was well-targeted with very little wasted effort
- **Query type features punch above their weight** — `is_numeric`, `is_description`, `is_entity` and their interactions all contribute positively in both analyses
- **`bigram_overlap` is the only consistent noise feature** — redundant given TF-IDF already captures n-gram overlap via `ngram_range=(1,2)`
- **`bm25_score` shows feature redundancy vs reliance** — ranked low in ablation (other features compensate during retraining) but shows genuine positive reliance after training (+0.0036), a known phenomenon in feature importance analysis

![](images/feature_ablation.png "Feature Ablation")

### Parsimonious Model
Guided by permutation importance, we trained a reduced model using only the 7 most informative features:

```python
PARSIMONIOUS_FEATURES = [
    'tfidf_cosine_sim',       # dominant feature by far (+0.1173 permutation)
    'query_term_coverage',    # strong lexical signal (+0.0216)
    'passage_has_number',     # numeric content is discriminative (+0.0211)
    'idf_weighted_coverage',  # weighted matching signal (+0.0193)
    'is_numeric',             # query type signal (+0.0183)
    'passage_position',       # structural position signal (+0.0135)
    'both_have_number',       # numeric content interaction (+0.0124)
]
```

| Metric | Enriched (22 features) | Parsimonious (7 features) | Δ |
|--------|----------------------|--------------------------|---|
| NDCG@10 | **0.6139** | 0.6026 | ↓0.0113 |
| MAP | **0.4899** | 0.4751 | ↓0.0148 |
| MRR | **0.4951** | 0.4802 | ↓0.0149 |

The parsimonious model achieved competitive performance but consistently underperformed across all metrics, suggesting the 15 additional features — while individually weak — provide small collective gains. The enriched model was retained as the final model.

---

## 6. Model Results

### Experiment Comparison

| | Baseline (9 features) | Enriched (22 features) |
|--|----------------------|----------------------|
| **Architecture** | (128, 64, 32) | (128, 64, 32) |
| **Optimizer** | Adam | Adam |
| **Dropout** | 0.3 | 0.3 |
| **Train Loss** | 0.3184 | **0.3138** |
| **Val Loss** | 0.3217 | **0.3187** |
| **Train AUC** | 0.6858 | **0.7068** |
| **Val AUC** | **0.6975** | 0.6894 |
| **NDCG@5** | 0.5230 | **0.5422** |
| **NDCG@10** | 0.6021 | **0.6139** |
| **MAP** | 0.4744 | **0.4899** |
| **MRR** | 0.4809 | **0.4951** |
| **Precision@5** | 0.1610 | **0.1657** |
| **Precision@10** | 0.1060 | 0.1060 |

### Final Model — Test Set Evaluation

| Metric | Score |
|--------|-------|
| **NDCG@5** | 0.5422 |
| **NDCG@10** | 0.6136 |
| **MAP** | 0.4897 |
| **MRR** | 0.4945 |
| **Precision@5** | 0.1657 |
| **Precision@10** | 0.1062 |
| **Test Loss** | 0.3145 |
| **Test AUC** | 0.6977 |

### Interpretation

- **NDCG@10 of 0.61** is solid for a tabular-only model with no neural text embeddings. Production neural retrieval systems (BM25 + BERT reranker) achieve ~0.70+ on MS MARCO, establishing a clear ceiling for future work.

- **MRR of 0.49** means on average the first relevant passage appears around rank 2 — the model is consistently placing relevant content near the top.

- **Train AUC improved from 0.686 → 0.707** with enriched features, indicating the model had more signal to learn from and continued improving throughout all 20 epochs rather than plateauing early.

### Performance Trajectory
```
Baseline DeepFM  (9 features)    →  NDCG@10: 0.6021, MAP: 0.4744
Enriched DeepFM  (22 features)   →  NDCG@10: 0.6136, MAP: 0.4897  (Final model)
Parsimonious     (7 features)    →  NDCG@10: 0.6026, MAP: 0.4751
Future work: SBERT embeddings    →  Expected NDCG@10: 0.65+
```

### Limitations & Future Work
- **Feature ceiling**: all model variants plateau around NDCG@10 ~0.61, suggesting the bottleneck is feature quality rather than architecture or optimization
- **Dense semantic embeddings**: adding sentence-BERT (e.g. `all-MiniLM-L6-v2`) cosine similarity as a feature would provide richer semantic signal beyond TF-IDF
- **Listwise training**: the current setup uses pointwise binary labels; pairwise or listwise loss functions (e.g. LambdaRank) could better optimize ranking directly
- **Larger dataset**: using the full MS MARCO training split (~500k queries) rather than a 10k validation sample

---

## 7. Reproducibility

All results are fully reproducible. Run cells in order from top to bottom using **Kernel → Restart & Run All**.

Key reproducibility measures:
- `set_seed(42)` applied before every model instantiation
- All splits use `random_state=42`
- Splits are by `query_id` to prevent query group leakage
- `torch.save(model, ...)` saves the full model object (not just state dict) for reliable reloading

```python
# Reload saved model
model = torch.load('../models/deepfm_final.pt', map_location=device, weights_only=False)
model.eval()
```

---

## 8. Repository Structure

```
├── notebooks/
│   ├── 1_feature_engineering.ipynb
│   └── 2_model_training.ipynb
│
├── models/
│   ├── deepfm_final.pt          # Full model object
    ├── deepfm_best.pt           # Full model object
│   ├── deepfm_parsimonious.pt   # Parsimonious model
│   └── model_config.json        # Architecture config
│
├── figures/
│   ├── training_curves.png
│   ├── ablation_results.png
│   └── permutation_importance.png
│
├── .gitignore
└── README.md
```

> **Note**: Pickle files (`*.pkl`) are excluded from version control via `.gitignore`. Pre-computed feature files are shared via Google Drive. Contact the team for access.

---

## 9. Setup & Installation

```bash
pip install datasets pandas numpy scikit-learn torch deepctr-torch rank-bm25 tqdm sentence-transformers cloudpickle
```

### Quick Start

```python
# Load pre-computed features (skip feature engineering)
import pandas as pd
df = pd.read_pickle('df_features_enriched.pkl')

# Load final trained model
import torch
model = torch.load('models/deepfm_final.pt', map_location='cpu', weights_only=False)
model.eval()

# Predict
test_preds = model.predict(test_input, batch_size=256).flatten()
```

---

## References

- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247). IJCAI.
- Nguyen, T., et al. (2016). [MS MARCO: A Human Generated Machine Reading Comprehension Dataset](https://arxiv.org/abs/1611.09268).
- Shen W. (2020). [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch).

---

