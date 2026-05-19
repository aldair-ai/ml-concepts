# Concept Map — DSC 240 Machine Learning

How every topic connects to every other, and to the broader goal of statistical learning theory.

---

## The Central Problem

```
     What we want:      f* = argmin E_{unseen} L(f(x), y)
     What we compute:   f_ERM = argmin_{f ∈ H} (1/n) Σ L(f(xi), yi)

     The gap:           E[L(f_ERM, y)] - L(f*) 
     Controlled by:     (1) Expressiveness of H  +  (2) Generalization
```

---

## Topic Dependencies

```
Probability & Statistics (prereq)
    │
    ├─► IID assumption ──────────────────────► Statistical model for data
    │
    ├─► Bayes Rule ──────────────────────────► Bayes Optimal Classifier
    │       └─ P(y|x) = P(x|y)P(y)/P(x)           └─ c*(x) = argmax P(y|x)
    │                                                └─ Bayes error ε* (floor)
    │
    └─► Law of Large Numbers ────────────────► Hoeffding Inequality
            └─ X̄_n → μ                              └─ |L̂_n - L| ≤ √(log/2n)

Linear Algebra (prereq)
    │
    ├─► Inner products ──────────────────────► Perceptron
    │       └─ ⟨w,x⟩ = sign decision               └─ T ≤ (M/m)²  (Novikoff)
    │
    ├─► Convex optimization ─────────────────► SVM
    │       └─ min ‖w‖², QP                          └─ max-margin separator
    │                                                └─ soft margin (slack ξ)
    │
    └─► Norms & distances ───────────────────► k-NN classifier
                                                    └─ Cover-Hart: err ≤ 2ε*
```

---

## The Generalization Tower

```
Single function h
    └─► Hoeffding:   P(|L̂ - L| ≥ t) ≤ 2e^{-2nt²}

Finite class H with |H| = M
    └─► Union bound: P(∃h: |L̂ - L| ≥ t) ≤ 2M·e^{-2nt²}
    └─► Bound:       L ≤ L̂ + √(log(2M/δ)/2n)

Infinite class H
    └─► VC theory:   L(c_ERM) ≤ L_emp + √(h(log(2n/h)+1) - log(ε/4))/n
    └─► Key concept: VC dimension = size of largest shattered set
    └─► Rademacher, covering numbers, margin bounds (extensions)
```

---

## The Classical vs. Modern Divide

```
Classical ML (pre-2012)
├─ Bias-variance tradeoff
├─ U-shaped test error curve  
├─ Goal: find "sweet spot" (model selection, regularization)
└─ Theory: VC bounds, SRM principle (Vapnik 1998)

Modern ML (2012–present)
├─ Interpolation: train error → 0 
├─ "Double descent" risk curve
├─ Very overparameterized models work BETTER
└─ Theory: benign overfitting, min-norm solutions, kernel methods

Key insight (Belkin et al., PNAS 2019):
    Classical curve ends WHERE modern ML starts.
    VC bounds are vacuous for overparameterized models — 
    new theory needed (still active research area!).
```

---

## Connection to DSC 215 (Science Reliability)

Every failure mode from the Ioannidis/Merton framework has an ML analog:

| DSC 215 (Scientific failure)    | DSC 240 (ML analog)                          |
|---------------------------------|----------------------------------------------|
| No control group                | No held-out test set                         |
| Motivated reasoning             | Choosing H after seeing test data            |
| File drawer (negative results)  | Reporting only best hyperparameter runs      |
| Underpowered studies            | Too little training data (large Hoeffding gap)|
| Multiple comparisons            | Testing many models without correction        |
| Unfalsifiable claims            | Vacuous VC bounds (always ≥ 1)               |
| Replication failure             | Model fails on new distribution (OOD)        |
| Mertonian: Disinterestedness    | Financial conflicts → cherry-picked benchmarks|

**Core unifying principle:**  
The gap between *empirical risk* (what you see on training data)  
and *expected risk* (what you get on new data)  
is the central quantity in both statistical inference and ML.

---

## Algorithms Summary Table

| Algorithm   | Type    | Training   | Hypothesis class    | Convergence          | Key parameter |
|-------------|---------|------------|---------------------|----------------------|---------------|
| k-NN        | Instance| None       | All of ℝⁿ (memory)  | Cover-Hart: ≤ 2ε*    | k             |
| Perceptron  | Linear  | Online     | Halfspaces          | T ≤ (M/m)² if sep.   | margin m      |
| SVM         | Linear  | Batch (QP) | Halfspaces          | Exact (convex)       | C (slack)     |
| Logistic    | Linear  | Batch (GD) | Halfspaces          | Convergent           | λ (reg.)      |
| Neural Net  | Nonlin  | SGD        | Turing-complete     | Empirically good     | arch, lr      |
| Kernel SVM  | Nonlin  | Batch (QP) | RKHS                | Exact (convex)       | C, kernel     |

---

## Open Questions (Active Research)

1. **Why does interpolation work?** Min-norm bias? Implicit regularization of SGD?
2. **Adversarial examples:** Directly connected to interpolation — are asymptotically dense.
3. **Does the double descent curve explain LLM behavior?** Scaling laws.
4. **New generalization theory:** VC bounds are too loose for deep nets. What replaces them?
