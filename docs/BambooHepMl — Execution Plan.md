# BambooHepMl — Execution Plan

> This document translates **requirements.md** and **rule.md** into a concrete, end‑to‑end **execution plan**.
> It is designed to be directly consumable by an autonomous agent or human developer.
> All milestones, tasks, and acceptance criteria are **binding** unless explicitly revised.

------

## 0. Context & Intent

BambooHepMl is an **engineering‑grade HEP + ML pipeline**, not a demo or research notebook.

The goal of this plan is to:

- Replace mock‑driven infrastructure with **real EDM4hep / podio ROOT workflows**
- Establish a **single, correct, reproducible pipeline** from ROOT → features → training → evaluation → inference
- Enable **classification and regression tasks** on real detector hit data
- Provide a foundation that is **scalable, auditable, and physically meaningful**

This plan assumes:

- `rule.md` is enforced
- Development is done under `conda activate bambooHepml_beta`
- Mock code is disposable if it conflicts with real ROOT handling



------

## 1. High‑Level Architecture (Target State)

```
ROOT (EDM4hep / podio)
   ↓
ROOT Reader (uproot + awkward)
   ↓
FeatureGraph
   ↓
Feature Reduction (event‑level scalars)
   ↓
Dataset (train / val / test)
   ↓
Model
   ├─ Torch (MLP / FNN)
   └─ XGBoost (numpy path)
   ↓
Evaluator
   ↓
MLflow Tracker
   ↓
Predictor
   ↓
ROOT Output (TTree: pred)
```

Single source of truth:

- Real ROOT structure
- Explicit FeatureConfig
- Logged experiment metadata

------

## 2. Milestone Overview

| Milestone | Name                     | Objective                             |
| --------- | ------------------------ | ------------------------------------- |
| M0        | Data Foundation          | Real ROOT ingestion + feature mapping |
| M1        | Classification Pipeline  | 3‑class end‑to‑end training + eval    |
| M2        | Regression Pipeline      | Incident energy regression            |
| M3        | XGBoost Integration      | Non‑torch model path                  |
| M4        | MLflow & Reproducibility | Full experiment governance            |
| M5        | Prediction Pipeline      | Standalone inference + ROOT output    |

Each milestone must pass **acceptance criteria** before proceeding.

------

## 3. M0 — Data Foundation (Critical)

### Objective

Build a **real‑data‑driven data layer** that correctly ingests EDM4hep ROOT files and produces deterministic event‑level features.

### Tasks

#### M0.1 ROOT Reader Refactor

- Implement or refactor ROOT reader using `uproot` + `awkward`
- Explicitly support:
  - podio collection branches
  - variable‑length arrays per event
- Hard‑bind to real branch names (e.g. `EcalBarrelCollection.energy`)

#### M0.2 FeatureConfig Redesign

- Rewrite FeatureConfig to cover **exactly 23 features** (no MC*)
- For each feature, declare:
  - source branch
  - data type (ragged)
  - reduction rule

#### M0.3 Feature Reduction Implementation

Reduction rules (mandatory):

- Energy / EDep → `sum()`
- Position → energy‑weighted mean

All reductions must:

- Be explicit
- Be unit‑tested on real ROOT events

#### M0.4 Dataset Construction

- Build Dataset abstraction with:
  - explicit train / val / test split
  - deterministic ordering
- Output format:
  - numpy arrays (features)
  - numpy arrays (labels)

### Acceptance Criteria

- Can load all 3 ROOT files without error
- Produces a numeric feature matrix of shape `(N_events, 23)`
- Feature values are physically sensible (non‑NaN, non‑empty)

------

## 4. M1 — Classification Pipeline

### Objective

Run a **full 3‑class classification pipeline** on real ROOT data.

### Tasks

#### M1.1 Label Mapping

- Implement `label:path` mapping via config
- Auto‑generate:
  - integer label
  - one‑hot labels

#### M1.2 Torch Model Training

- Implement baseline MLP / FNN classifier
- Fixed seeds
- CPU / GPU / MPS support

#### M1.3 Evaluation

- Implement evaluation metrics:
  - ROC curves
  - confusion matrix
  - score distributions

#### M1.4 ROOT Output Writer

- Implement `write_root_output()` using `uproot.recreate()`
- Output `TTree: pred` with schema defined in Master Requirements

### Acceptance Criteria

- End‑to‑end training runs on real data
- Evaluation plots are generated and saved
- ROOT output file contains valid `pred` tree

------

## 5. M2 — Regression Pipeline

### Objective

Predict `_IncidentEnergy` using the same feature pipeline.

### Tasks

#### M2.1 Target Mapping

- Map `_IncidentEnergy` as regression target
- Handle tree version explicitly (`events;6`)

#### M2.2 Regression Model

- Implement baseline MLP regressor

#### M2.3 Regression Evaluation

- Compute:
  - residual
  - bias
  - resolution
  - pull (if sigma available)

### Acceptance Criteria

- Regression training converges
- Plots are physically reasonable
- ROOT output matches required schema

------

## 6. M3 — XGBoost Integration

### Objective

Enable non‑torch models without feature duplication.

### Tasks

- Reuse FeatureGraph + Dataset
- Implement XGBoost trainer consuming numpy features
- Log model and metrics to MLflow

### Acceptance Criteria

- XGBoost produces comparable outputs
- No forked feature pipeline exists

------

## 7. M4 — MLflow & Reproducibility

### Objective

Make experiments fully traceable and reproducible.

### Tasks

- Log:
  - full config YAML
  - feature definitions
  - metrics
  - plots
  - model artifacts
- Enforce seed control

### Acceptance Criteria

- Entire experiment can be replayed from MLflow

------

## 8. M5 — Prediction Pipeline

### Objective

Provide a standalone inference pipeline.

### Tasks

- Decouple Predictor from Trainer
- Load trained model + FeatureConfig
- Run inference on new ROOT files
- Write ROOT output (`pred` tree)

### Acceptance Criteria

- Inference works without training code
- Output ROOT file is valid and non‑destructive

------

## 9. Out‑of‑Scope (Explicit)

- Sequence / attention models
- Object‑level features
- Graph neural networks
- Online inference

------

## 10. Success Definition

The project is considered **successful** when:

- All milestones M0–M5 pass acceptance criteria
- No mock data is required to run the full pipeline
- Results are reproducible, inspectable, and physically meaningful

------

**Status**: Ready for Execution