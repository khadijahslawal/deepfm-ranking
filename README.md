# DeepFM for Search Relevance Ranking

DeepFM for Search Relevance Ranking applied to Microsoft MARCO passage ranking data

University of Chicago — MSc in Applied Data Science

## Model Description & Applications

DeepFM (Deep Factorization Machine) is a neural architecture that jointly trains:
	•	A Factorization Machine (FM) component
	•	A Deep Neural Network (DNN) component

over shared input embeddings. Introduced by Guo et al. (2017), DeepFM learns both low-order and high-order feature interactions end-to-end, without manual feature engineering.
This architecture is widely used in:
	•	Click-through rate (CTR) prediction
	•	Search relevance ranking
	•	Recommendation systems
	•	Advertising ranking systems


## Key Components & Mathematical Formulation

### Input Representation

Let the input feature vector be:

$$
\mathbf{x} \in \mathbb{R}^d
$$

where:
	•	$d$ = total number of features
	•	Features may be dense (continuous) or sparse (categorical / one-hot encoded)
	•	Each feature field $i$ has an embedding vector

$$
\mathbf{v}_i \in \mathbb{R}^k
$$

where $k$ is the embedding dimension (typically 4–16).

The input vector is:

$$
\mathbf{x} = [x_1, x_2, \ldots, x_d]
$$


### Factorization Machine (FM) Component

The FM component models:
	•	First-order (linear) effects
	•	Second-order (pairwise) feature interactions


First-Order (Linear) Term
$$
y_{\text{order1}} = w_0 + \sum_{i=1}^{d} w_i x_i
$$

where:
	•	$w_0$ = global bias
	•	$w_i$ = feature weight


Second-Order (Pairwise) Interaction Term
$$
y_{\text{order2}} =
\sum_{i=1}^{d} \sum_{j=i+1}^{d}
\langle \mathbf{v}_i, \mathbf{v}_j \rangle \cdot x_i x_j
$$

where the dot product between embeddings is:

$$
\langle \mathbf{v}i, \mathbf{v}j \rangle =
\sum{l=1}^{k} v{il} v_{jl}
$$

Efficient Computation Trick
To reduce computational complexity from $O(d^2)$ to $O(kd)$:

$$
y_{\text{order2}} = \frac{1}{2}
\sum_{l=1}^{k}
\left(
\left(
\sum_{i=1}^{d} v_{il} x_i
\right)^2
\sum_{i=1}^{d} v_{il}^2 x_i^2
\right)
$$

Combined FM Output

$$
y_{\text{FM}} =
y_{\text{order1}} + y_{\text{order2}}
$$


### Deep Neural Network (DNN) Component

The DNN captures:
	•	Higher-order feature interactions
	•	Non-linear relationships
	•	Complex hierarchical patterns

The DNN takes the same shared embeddings as input and passes them through multiple fully connected layers:

$$
\mathbf{a}^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} \mathbf{a}^{(l)} + \mathbf{b}^{(l)} \right)
$$

where:
	•	$\sigma(\cdot)$ is a non-linear activation (e.g., ReLU)
	•	$\mathbf{W}^{(l)}$ and $\mathbf{b}^{(l)}$ are trainable parameters


Final Prediction

The final DeepFM prediction combines FM and DNN outputs:

$$
\hat{y} = \sigma \left( y_{\text{FM}} + y_{\text{DNN}} \right)
$$

where $\sigma$ is typically a sigmoid function for binary relevance prediction.
