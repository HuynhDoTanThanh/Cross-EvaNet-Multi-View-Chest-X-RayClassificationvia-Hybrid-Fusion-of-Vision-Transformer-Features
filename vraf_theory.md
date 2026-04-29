# View-Role Attention Fusion (VRAF): A Theory for CXR Multi-View Prediction

## 1. Core Theory: View-Role Decomposition

In clinical radiology, a chest X-ray reading proceeds through a structured **view-role protocol**. The radiologist does not treat multiple views as interchangeable — each view serves a fundamentally different *diagnostic role*:

| View                         |      Symbol      | Diagnostic Role                | What It Captures                                                                                                    |
| :--------------------------- | :--------------: | :----------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Frontal** (PA/AP)    | $\mathbf{z}_F$ | **Primary Assessment**   | Heart silhouette, lung fields, mediastinum, pleural margins, devices                                                |
| **Lateral**            | $\mathbf{z}_L$ | **Depth Disambiguation** | Retrocardiac space, posterior costophrenic angles, vertebral bodies, retrosternal space                             |
| **Cross-view** (fused) | $\mathbf{z}_X$ | **Synergistic Evidence** | Correlations invisible to either view alone — e.g., a frontal opacity that aligns with a lateral posterior density |

> [!IMPORTANT]
> **Key insight**: These three sources are NOT symmetric. The frontal view is the *primary diagnostic anchor*; the lateral view provides *complementary depth information*; the cross-view representation captures *synergistic correlations* that neither view encodes independently. An attention mechanism for CXR must respect this asymmetry.

---

## 2. Disease-View Affinity: The Clinical Prior

Each of the 14 pathologies in the CXR task has a known clinical relationship with specific views. We formalize this as a **Disease-View Affinity Matrix** $\mathbf{A}^* \in \mathbb{R}^{14 \times 3}$:

```
                        Frontal (F)   Lateral (L)   Cross-view (X)
                        ──────────    ──────────    ──────────────
Atelectasis              ●●●○          ●●○○          ●●○○
Cardiomegaly             ●●●●          ●○○○          ●○○○
Consolidation            ●●●○          ●●●○          ●●●○
Edema                    ●●●○          ●●○○          ●●○○
Enlarged Cardiomed.      ●●●●          ●●○○          ●●○○
Fracture                 ●●○○          ●●●○          ●●●●    ← needs multi-angle
Lung Lesion              ●●●○          ●●○○          ●●●○
Lung Opacity             ●●●○          ●●○○          ●●○○
No Finding               ●●○○          ●●○○          ●○○○
Pleural Effusion         ●●●○          ●●●●          ●●●○    ← posterior layering
Pleural Other            ●●○○          ●●●○          ●●●○
Pneumonia                ●●●●          ●●○○          ●●○○    ← frontal-dominant
Pneumothorax             ●●●●          ●○○○          ●○○○    ← visceral line
Support Devices          ●●●●          ●○○○          ●○○○
```

Three diagnostic patterns emerge:

1. **Frontal-dominant** ($\alpha_F \gg \alpha_L, \alpha_X$): Cardiomegaly, Pneumothorax, Pneumonia, Support Devices
   → The pathological sign is directly visible on PA/AP view
2. **Lateral-enhanced** ($\alpha_L$ elevated): Pleural Effusion, Fracture, Pleural Other, Consolidation→ The lateral view reveals posterior/retrocardiac structures obscured in frontal projection
3. **Cross-view-dependent** ($\alpha_X$ elevated): Fracture, Consolidation (retrocardiac), Lung Lesion
   → Diagnosis requires correlating evidence across both projections simultaneously

---

## 3. Architectural Design: View-Role Attention Fusion (VRAF)

### 3.1 Architecture Overview

```
Frozen Encoder                    Multi-View Encoder
    │                                    │
    ├── img1 → z_F (frontal)             │
    │                                    │
    ├── img2 → z_L (lateral)       P' → z_X (cross-view)
    │                                    │
    └─────────────┬──────────────────────┘
                  │
         ┌────────▼────────┐
         │   VIEW-ROLE     │
         │   ATTENTION     │
         │   FUSION (VRAF) │
         └────────┬────────┘
                  │
              ŷ' (correction)
                  │
    ŷ = ½(½(ŷ₁ + ŷ₂) + ŷ')
```

### 3.2 Formal Definition

**Step 1 — View-Role Token Sequence**

Each view feature is augmented with a learnable *role embedding* that encodes its diagnostic function:

$$
\mathbf{t}_F = \mathbf{z}_F + \mathbf{r}_F, \quad \mathbf{t}_L = \mathbf{z}_L + \mathbf{r}_L, \quad \mathbf{t}_X = \mathbf{z}_X + \mathbf{r}_X
$$

where $\mathbf{r}_F, \mathbf{r}_L, \mathbf{r}_X \in \mathbb{R}^D$ are learnable role embeddings (ℓ₂-normalized). These embeddings help the attention distinguish *why* each feature is present, not just *what* it contains.

The role-augmented tokens are stacked:

$$
\mathbf{T} = [\mathbf{t}_F;\; \mathbf{t}_L;\; \mathbf{t}_X] \in \mathbb{R}^{3 \times D}
$$

**Step 2 — Disease Query Prototypes**

A set of $C = 14$ learnable disease query vectors:

$$
\mathbf{Q} = [\mathbf{q}_1; \ldots; \mathbf{q}_C] \in \mathbb{R}^{C \times D}
$$

Each $\mathbf{q}_c$ acts as a prototype for pathology $c$. During training, each query *learns to attend to the views that are most informative for its associated disease*.

**Step 3 — Multi-Head Cross-Attention**

Each disease query attends over the 3 role-augmented view tokens:

$$
\text{head}_h = \text{softmax}\!\left(\frac{(\mathbf{Q}\mathbf{W}_h^Q)(\mathbf{T}\mathbf{W}_h^K)^\top}{\sqrt{d_k}}\right) \mathbf{T}\mathbf{W}_h^V
$$

$$
\mathbf{A} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\mathbf{W}^O
$$

The attention matrix $\boldsymbol{\alpha} \in [0,1]^{C \times 3}$ is the **Disease-View Affinity Matrix** — it directly tells us, for each pathology, how much it relies on the frontal, lateral, and cross-view representations.

**Step 4 — Gated Fusion with Conservative Initialization**

$$
\mathbf{g} = \sigma(\mathbf{W}_g[\mathbf{Q};\; \mathbf{A}] + \mathbf{b}_g)
$$

$$
\tilde{\mathbf{O}} = \mathbf{g} \odot \mathbf{A} + (1 - \mathbf{g}) \odot \mathbf{Q}
$$

Gate bias $\mathbf{b}_g$ initialized to $-2.0$ so that $\sigma(-2) \approx 0.12$, meaning the model starts by mostly relying on the disease query priors and gradually learns to trust the attention.

**Step 5 — Per-Label Prediction**

$$
\hat{y}'_c = \mathbf{w}^\top \text{LN}(\tilde{\mathbf{O}}_c) + b_c
$$

---

## 4. What Makes This CXR-Specific (vs. Generic Attention)

| Generic G-CAF         | CXR-Specific VRAF                                                            |
| :-------------------- | :--------------------------------------------------------------------------- |
| 3 anonymous tokens    | 3 role-named tokens: Frontal, Lateral, Cross-view                            |
| No role embeddings    | ℓ₂-normalized role embeddings encode diagnostic function                   |
| Symmetric treatment   | Asymmetric: frontal is the*anchor*, lateral/cross-view are *refinements* |
| Generic label queries | Disease prototypes that learn clinical affinity patterns                     |
| Opaque attention      | Interpretable$14 \times 3$ Disease-View Affinity Matrix                    |

### 4.1 Role Embeddings: Why They Matter

Without role embeddings, the attention mechanism sees three 768-dim vectors with no indication of their diagnostic origin. The model must learn *from scratch* that token[0] is frontal and token[2] is lateral — a difficult implicit learning task.

Role embeddings provide an explicit **diagnostic identity signal**:

- $\mathbf{r}_F$ learns to encode "I am the primary frontal assessment"
- $\mathbf{r}_L$ learns to encode "I provide depth/posterior information"
- $\mathbf{r}_X$ learns to encode "I capture cross-view synergies"

This is analogous to how your existing multi-view Transformer uses **view embeddings** ($\mathbf{v}_1, \mathbf{v}_2$) at the token-merging stage — but now applied at the fusion decision stage.

### 4.2 Asymmetric Design: Frontal as Anchor

In clinical practice, the frontal view is always read first. The lateral view is consulted to *resolve ambiguities* in the frontal reading. We encode this asymmetry by:

1. The role embeddings naturally capture this — $\mathbf{r}_F$ will learn different features than $\mathbf{r}_L$
2. The gating mechanism preserves the disease query prior when the lateral/cross-view don't add value
3. The outer boosting formula $\hat{\mathbf{y}} = \frac{1}{2}(\frac{1}{2}(\hat{\mathbf{y}}_1 + \hat{\mathbf{y}}_2) + \hat{\mathbf{y}}')$ already ensures the single-view predictions (primarily frontal) serve as the baseline

---

## 5. Interpretability: The Disease-View Affinity Matrix

The most powerful output of VRAF is the **learned $14 \times 3$ attention matrix** $\boldsymbol{\alpha}$. After training, you can:

### 5.1 Visualize per-disease view reliance

```
Pathology          | Frontal  Lateral  Cross-view
───────────────────┼─────────────────────────────
Cardiomegaly       | ██████░░ ░░░░░░░░ ██░░░░░░
Pleural Effusion   | ████░░░░ ██████░░ ████░░░░
Fracture           | ████░░░░ ████░░░░ ██████░░
Pneumothorax       | ████████ ░░░░░░░░ ░░░░░░░░
```

### 5.2 Validate against clinical knowledge

If Cardiomegaly's attention is heavily on frontal → ✅ clinically correct
If Pneumothorax attends heavily to lateral → ❌ something is wrong

This provides a **built-in sanity check** for your model.

### 5.3 Use for clinical explainability

For a specific patient case, you can report: *"The model diagnosed Pleural Effusion with 40% reliance on the frontal view, 45% on the lateral view, and 15% on cross-view correlation."*

---

## 6. Summary: Theory in One Paragraph

> **View-Role Attention Fusion (VRAF)** formalizes the clinical observation that chest X-ray diagnosis follows a *view-role protocol* where the frontal radiograph serves as the primary diagnostic anchor, the lateral projection provides depth disambiguation for retrocardiac and posterior pathologies, and cross-view feature fusion captures synergistic evidence invisible to either view alone. VRAF introduces role embeddings to encode each view's diagnostic function, and uses label-specific disease query prototypes that attend over the role-augmented view tokens via gated multi-head cross-attention. The resulting Disease-View Affinity Matrix $\boldsymbol{\alpha} \in [0,1]^{C \times 3}$ provides both improved label-specific prediction and direct clinical interpretability, enabling validation against established radiological knowledge.
