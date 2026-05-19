# Machine Learning Foundations
### DSC 240 · UCSD · Spring 2026

> *"More data ≠ more knowledge or insight."* — Mikhail Belkin

A structured study companion covering the theory and algorithms of modern machine learning — from Bayes-optimal classifiers to the double descent phenomenon. Built alongside **DSC 240** (Machine Learning) at UC San Diego.

---

## 📂 Repository Structure

```
ml-concepts-repo/
├── README.md                    ← You are here
├── notebooks/
│   ├── 01_bayes_classifier.ipynb        ← Optimal decision rules, class densities
│   ├── 02_perceptron.ipynb              ← Rosenblatt's algorithm + convergence
│   ├── 03_svm.ipynb                     ← Hard & soft margin, hinge loss
│   ├── 04_erm_generalization.ipynb      ← Hoeffding, union bound, VC theory
│   ├── 05_double_descent.ipynb          ← Interpolation & benign overfitting
│   └── 06_peer_review_ppv.ipynb         ← Science reliability (DSC 215 bridge)
├── demos/
│   ├── interactive_concepts.html        ← Browser-based visual playground
│   ├── perceptron_demo.py               ← Live perceptron training animation
│   └── knn_voronoi.py                   ← k-NN decision boundaries
├── docs/
│   └── concept_map.md                   ← How all topics connect
└── assets/
    └── figures/                         ← Generated plots
```

---

## 🧠 Concepts Covered

### Module 1 — What Is Learning?
The eternal problem of reconciling theory to observation, from Ptolemy's epicycles to neural networks. Learning is defined as finding patterns in data **predictive of unseen data**.

| Concept | Key Idea |
|---------|----------|
| Data as vectors | Images → pixel arrays → feature vectors in ℝⁿ |
| Supervised learning | Given (x₁,y₁)…(xₙ,yₙ), build c: ℝⁿ → {+1,−1} |
| Autoregressive models | Text generation = next-token prediction |
| IID assumption | (x,y) drawn from a fixed joint distribution |

### Module 2 — Optimal Classification
What does "best possible" even mean? The economics framing: **lose $1 per wrong prediction**, minimize expected loss.

**Bayes Optimal Classifier:**
```
c*(x) = +1  if  P(y=+1 | x) > P(y=-1 | x)
        -1  otherwise
```
Equivalently, compare class-conditional likelihoods:
```
P(x | y=+1) · P(+1)  vs.  P(x | y=-1) · P(-1)
```
This works because of **Bayes' rule** — and it defines the floor no algorithm can beat.

### Module 3 — The Perceptron
Frank Rosenblatt's 1957 algorithm, analyzed by Novikoff (1962). A simple online update rule that provably converges when data is linearly separable.

**Update rule:**
```python
if sign(<w, x>) != y:
    w = w + y * x   # nudge w toward correct side
```

**Convergence bound** (Novikoff's theorem):
```
T ≤ (M² · ‖w*‖²) / m²
```
where `m` is the **margin** and `M` bounds the data norm. The algorithm stops in finite steps — the tighter the margin, the more updates needed.

### Module 4 — Support Vector Machines
SVMs find the **maximum-margin** separating hyperplane. Hard margin → convex quadratic program. Soft margin → adds slack variables ξᵢ for non-separable data.

**Soft-margin primal:**
```
min  ½‖w‖² + C·Σξᵢ
 w,b,ξ
s.t.  yᵢ(⟨w,xᵢ⟩ + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0
```
This is equivalent to minimizing **regularized hinge loss**:
```
½‖w‖² + C · Σ max(0, 1 − yᵢf(xᵢ))
```

### Module 5 — Empirical Risk Minimization (ERM)
The unifying framework for supervised learning.

**Goal of ML:** minimize expected loss on unseen data  
**Goal of ERM:** minimize empirical loss on training data, within hypothesis class ℋ

Two requirements for ERM to work:
1. Empirical loss ≈ expected loss for all f ∈ ℋ  *(generalization)*
2. ℋ contains a good approximation to f*  *(expressiveness)*

**Common loss functions:**

| Name | Formula | Used in |
|------|---------|---------|
| Square loss | (c(x) − y)² | Regression, neural nets |
| Hinge loss | max(0, 1 − yc(x)) | SVM |
| Logistic loss | ln(1 + e^{−yc(x)}) | Logistic regression |
| 0-1 loss | 𝟙[c(x) ≠ y] | True error (NP-hard to minimize) |

### Module 6 — Generalization Theory
How do we know a model that fits training data will generalize?

**Hoeffding inequality** (single function):
```
P(|L̂ₙ(h) − L(h)| ≥ t) ≤ 2e^{−2nt²}
```
With probability ≥ 1−δ:
```
L(h) ≤ L̂ₙ(h) + √(log(2/δ) / 2n)
```

**Union bound over M classifiers:**
```
L(h) ≤ L̂ₙ(h) + √(log(2M/δ) / 2n)
```

**VC bound** (infinite hypothesis classes):
```
L(cₑᵣₘ) ≤ Lₑₘₚ(cₑᵣₘ) + √(h(log(2n/h)+1) − log(ε/4)) / n)
```
The VC dimension `h` measures how "complex" a hypothesis class is — specifically, the size of the **largest set it can shatter**.

### Module 7 — The Double Descent Phenomenon
Classical wisdom: bias-variance tradeoff → U-shaped test error → find the "sweet spot".

**Modern reality:** very overparameterized models that *interpolate* training data can outperform classical models.

```
Classical regime:            Modern (interpolating) regime:
under-fit → sweet spot → overfit → [interpolation threshold] → keeps improving
```

Key results:
- Interpolating classifiers with *singular* kernels (e.g., Laplace) are **statistically optimal** [Belkin et al., NeurIPS 2018]
- Adversarial examples are an *asymptotically dense* consequence of interpolation
- Near-interpolation (very low train loss) is empirically optimal across speech, vision, and language tasks

---

## 🔗 Connection to DSC 215 — Science Reliability

These ML algorithms exist within a broader scientific ecosystem. The same failure modes that break scientific inference break ML:

| Scientific failure (DSC 215) | ML analog (DSC 240) |
|-----------------------------|---------------------|
| No control group | No test set / train=test evaluation |
| Motivated reasoning | Cherry-picking hypothesis class ℋ |
| File drawer problem | Reporting only successful runs |
| Unfalsifiable claims | Generalization bounds that are always vacuous |
| Replication failure | Model fails on new data distribution |

Both courses converge on: **the gap between what you see (empirical risk) and what you get (expected risk)** is the central problem.

---

## 📓 Notebook Summaries

### `01_bayes_classifier.ipynb`
- Simulate 2D Gaussian class-conditional densities
- Visualize the Bayes decision boundary
- Compare Parzen window estimation to the true boundary
- Measure error vs. Bayes risk

### `02_perceptron.ipynb`
- Implement the Perceptron from scratch
- Animate weight updates on linearly separable 2D data
- Verify Novikoff's convergence bound empirically
- Show failure on non-separable data

### `03_svm.ipynb`
- Implement soft-margin SVM using `cvxpy`
- Visualize support vectors and margin
- Compare hinge loss vs. logistic loss
- Tune C and observe bias-variance tradeoff

### `04_erm_generalization.ipynb`
- Demonstrate Hoeffding bound in simulation
- Compute VC dimension for linear classifiers
- Show how generalization gap shrinks with n
- Union bound: more classifiers → looser bound

### `05_double_descent.ipynb`
- Reproduce the double descent curve with random ReLU features
- Show classical U-shape ending at interpolation threshold
- Demonstrate benign overfitting with Laplace kernel
- Compare to early-stopping baseline

### `06_peer_review_ppv.ipynb`
- Ioannidis PPV model with Mertonian norm violations
- Simulate how bias u and power γ affect published truth rates
- Citation power-law (Lotka's Law) and Lorenz curve
- Peer review as noisy binary classifier

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/ml-concepts-repo
cd ml-concepts-repo
pip install numpy scipy matplotlib scikit-learn cvxpy jupyterlab
jupyter lab
```

Open any notebook in `notebooks/` or launch the interactive demo:
```bash
open demos/interactive_concepts.html
```

---

## 📚 References

- Belkin, M. (2021). *Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation.* Acta Numerica.
- Belkin, M., Hsu, D., Ma, S., Mandal, S. (2019). *Reconciling modern machine-learning practice and the classical bias-variance trade-off.* PNAS.
- Cover, T., Hart, P. (1967). *Nearest neighbor pattern classification.* IEEE Transactions on Information Theory.
- Novikoff, A. (1962). *On convergence proofs for perceptrons.*
- Vapnik, V. (1998). *Statistical Learning Theory.* Wiley.
- Zhang, C. et al. (2017). *Understanding deep learning requires rethinking generalization.* ICLR.
- Ioannidis, J.P.A. (2005). *Why Most Published Research Findings Are False.* PLOS Medicine.
- Ritchie, S. (2020). *Science Fictions.* Metropolitan Books.

---

## 👤 Author

**Aldair Espinoza** · DSC 240 · UC San Diego · Spring 2026  
Instructor: Prof. Mikhail Belkin

*This repo is a living document — updated as the course progresses.*
