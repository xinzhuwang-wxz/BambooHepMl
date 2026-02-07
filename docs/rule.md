# BambooHepMl — Project Rules & Conventions

> This document defines **non-negotiable rules, constraints, and conventions** for the BambooHepMl project.
> Its purpose is to prevent architectural drift, mock-driven design, and physics-invalid shortcuts.
> All contributors (human or agent) must follow these rules.

------

## 0. Prime Directive

**Reality First. Physics First. Pipeline Second.**

- The **single source of truth** is the *real ROOT data* and the *actual physics task*.
- Framework abstractions must adapt to data — **never the opposite**.
- A working end-to-end pipeline on real data is always preferred over a theoretically elegant but unvalidated design.

------

## 1. Environment Rules

- All development, training, evaluation, and inference **must be executed under**:

```bash
conda activate bambooHepml_beta
```

- No silent fallback to system Python or alternative environments.
- Environment-specific behavior must be explicit and documented.

------

## 2. Data Source & ROOT Handling Rules

### 2.1 Real ROOT Is Mandatory

- The pipeline **must be built and validated against real EDM4hep / podio ROOT files**.
- Mock datasets, synthetic Jet/MET branches, or placeholder structures:
  - **Must not influence architecture decisions**
  - May be deleted, replaced, or heavily rewritten

> Mock code exists only as thin sanity checks — never as design drivers.

### 2.2 ROOT Structure Assumptions

- Input data consists of:
  - `TTree: events`
  - podio-style variable-length collections
- Branches such as `EcalBarrelCollection.energy`, `TPCCollection.EDep`, etc. are **variable-length arrays per event**.
- There are **no true scalar physics features** in the input feature set.

------

## 3. Feature Rules (Critical)

### 3.1 Feature Scope

- **MC-derived branches (`MC\*`) are explicitly excluded** from all tasks.
- Only calorimeter and tracker hit–level collections are used as input features.

### 3.2 Feature Reduction (Baseline Phase)

- **All features must be reduced to event-level scalars.**
- **No padded arrays, no sequence tensors, no attention models** in the baseline pipeline.

Reduction strategies:

| Feature Type     | Rule                                |
| ---------------- | ----------------------------------- |
| Energy / EDep    | `sum()` per event                   |
| Position (x/y/z) | Energy-weighted mean: `Σ(x·E) / ΣE` |

- Every reduction **must be explicitly declared** in `FeatureConfig`.
- Silent padding, flattening, or implicit aggregation is forbidden.

### 3.3 Future Extensions

- Object-level / sequence representations are **Phase-2 features**.
- They must not leak into baseline abstractions or APIs.

------

## 4. Dataset & Pipeline Rules

- Dataset construction must be:
  - Deterministic
  - Reproducible
  - Explicit about train/val/test splits
- Feature extraction is performed **once** via a unified `FeatureGraph`.
- Downstream components (Torch, XGBoost, inference) consume the same feature outputs.

------

## 5. Model Rules

### 5.1 Torch Models

- MLP / FNN are treated as standard feed-forward networks.
- No model is allowed to re-interpret feature semantics.

### 5.2 XGBoost (Mandatory Integration Pattern)

- XGBoost **must share FeatureGraph + DataConfig** with Torch models.
- XGBoost:
  - Bypasses Torch `Trainer` and `DataLoader`
  - Directly consumes numpy feature matrices

> Separate feature pipelines for XGBoost are forbidden.

------

## 6. Task Definitions

### 6.1 Classification

- Multi-class classification (e.g. 3 energy classes).
- Labels are derived from file-path-to-label mappings.
- Outputs must include:
  - `label` (argmax)
  - `is_<class>` (one-hot)
  - `score_<class>` (softmax probabilities)

### 6.2 Regression

- Target: `_IncidentEnergy` (scalar float).
- Regression metrics must include:
  - prediction
  - residual
  - pull (if uncertainty is defined)

------

## 7. Output Rules

- Prediction results are written to ROOT using:
  - `uproot.recreate()`
  - `WritableDirectory`
- Output format:
  - Separate `TTree: pred`
  - **Never overwrite or modify the original `events` tree**

------

## 8. Evaluation Rules

- Evaluation must be automatic and reproducible.

Required outputs:

- **Classification**: ROC, confusion matrix, score distributions
- **Regression**: resolution, bias, pull distributions
- Sanity-check plots are mandatory artifacts

------

## 9. MLflow & Reproducibility Rules

- MLflow tracking is mandatory.
- Logged items must include:
  - Full config (YAML)
  - Feature definitions
  - Metrics
  - Plots
  - Model artifacts
- Random seeds must be fixed and recorded.

------

## 10. Anti-Patterns (Explicitly Forbidden)

- Designing around mock data instead of real ROOT
- Implicit feature aggregation
- Multiple feature pipelines per model type
- Padding/sequences introduced "just in case"
- Torch-specific assumptions leaking into non-Torch models
- Modifying physics meaning to satisfy framework constraints

------

## 11. Change Policy

- Any change that violates these rules requires:
  1. Explicit justification
  2. Update to this document
  3. Re-validation on real ROOT data

Until then, **these rules are binding**.

------

**Status**: Active / Enforced