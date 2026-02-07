

-------

# **需求和项目内容**

# BambooHepMl — requirement_plan

## 1. 项目背景与目标

BambooHepMl 是一个**工程级的高能物理 + 机器学习项目**，目标是在本地（macOS / Linux）以及可扩展环境中，稳定、可复现实验地完成 **ROOT → Dataset → 训练 → 评估 → 推理** 的完整流水线。

本项目在设计与实现上 **参考并吸收** 以下两个项目的优点：

- **/Users/bamboo/Desktop/weaver-core**：HEP 数据结构化、可扩展 Dataset 设计、训练/推理解耦
- **Users/bamboo/Desktop/Made-With-ML**：工程化配置管理、实验可复现性、MLflow 监控

当前 BambooHepMl 已具备基础框架，但在 **数据建模、配置治理、pipeline 严谨性与可扩展性** 上仍需系统性完善与重构。

> ⚠️ **强制约束**：
>
> - 所有实验必须在 `conda activate bambooHepml_beta` 环境下完成
> - 不允许为“跑通”而做 mock / shortcut，所有步骤必须物理与工程上合理
> - 更具体的约束请看rule.md

## 2. 输入数据与物理语义约束

### 2.1 输入 ROOT 文件

- 数据位置：`BambooHepMl/testdata/`
- 当前提供 3 个 ROOT 文件：
  - `sim_pi-_3.00GeV_40mmGS_1.root`
  - `sim_pi-_5.00GeV_40mmGS_1.root`
  - `sim_pi-_7.00GeV_40mmGS_1.root`

每个 ROOT 文件包含：

- `TTree: events`
- 约 1000 events
- EDM4hep / podio 风格分支（变长 array + object collection）

ROOT 内容示例（已验证）：

- calorimeter hit energy
- position (x, y, z)
- MCParticle momentum / vertex

## 3. 特征定义（Feature Contract）

### 3.1 统一特征列表（23 个）

#### 能量相关（15）

- EcalBarrelCollection.energy
- HcalBarrelCollection.energy
- EcalEndcapsCollection.energy
- HcalEndcapsCollection.energy
- EcalEndcapRingCollection.energy
- HcalEndcapRingCollection.energy
- EcalBarrelContributionCollection.energy
- HcalBarrelContributionCollection.energy
- LumicalCollection.energy
- TPCCollection.EDep
- TPCLowPtCollection.EDep
- TPCSpacePointCollection.EDep
- VXDCollection.EDep
- ITKBarrelCollection.EDep
- ITKEndcapCollection.EDep

#### 位置相关（8）

- EcalBarrelCollection.position.x
- EcalBarrelCollection.position.y
- EcalBarrelCollection.position.z
- HcalBarrelCollection.position.x
- HcalBarrelCollection.position.y
- EcalEndcapsCollection.position.z
- HcalEndcapsCollection.position.z
- TPCCollection.position.z

### 3.2 特征处理基本原则

- 明确区分：
  - **scalar**（单值）
  - **ragged / variable-length array**
- 所有裁剪（top-k / sum / mean / histogram）必须显式声明
- 新特征构建（如 sumE、weighted position）需可配置

## 4. 任务定义（Tasks）

### 4.1 多分类任务（Classification）

**目标**：

- 使用上述 23 个变量
- 进行 **3 分类问题**

**数据组织规则**：

- 测试的这三个，每个 ROOT 文件 = 一个 class
- 支持：
  - 每一类传入多个 ROOT 文件
  - 自定义 label 名称（如 `pi_3GeV`）

**输出要求**：

- 自动生成：
  - `is_label`
  - `label_score`（softmax / probability）
- 写入 output ROOT / parquet / npz（output还是tree吧，包含label和其score的branch以及观察变量的branch）

### 4.2 回归任务（Regression）

**目标**：

- 输入：
  - 上述 23 个变量作为feature
  - 回归目标是branch： `_IncidentEnergy`（float）

## 5. Pipeline 端到端要求

### 5.1 数据层（Data Layer）

- ROOT reader：
  - 支持 podio / edm4hep
  - 明确 mapping：ROOT → numpy / awkward
- Dataset 抽象：
  - train / val / test 明确划分
  - 支持 lazy loading

### 5.2 特征工程层

- variable-length 处理策略可配置
- 新变量构建模块化
- 所有变换可追溯（记录到 MLflow）

### 5.3 训练层

- 支持：
  - CPU / GPU / MPS
- 模型接口统一（sklearn / torch / xgboost 可扩展），测试网络是简单m l p, 简单fnn, xgboost
- 随机种子全链路固定

### 5.4 实验管理

- MLflow 必须接入

  ：

  - config
  - metrics
  - artifacts
  - model

### 5.5 评估与诊断

- 分类：ROC、confusion matrix、score 分布
- 回归：resolution、bias、pull
- 自动生成 sanity check plots

### 5.6 推理与产物

- pred pipeline 与 train 解耦
- 输出格式统一

## 6. 配置与工程规范

- 所有实验 **必须配置化**（YAML / Hydra 风格）
- 禁止：
  - 配置散落在代码中
  - magic number
- 目录结构需：
  - 清晰区分 data / model / experiment / output

## 7. 质量标准（Non-Functional Requirements）

- 可复现
- 可扩展
- 可维护
- 物理语义正确
- 工程风格对标成熟 ML 项目

## 8. 协作方式（Agent Interaction Contract）

- 本文档作为 **Master Requirement**
- 后续流程：
  1. 与智能体逐条澄清 /补充
  2. 拆分为 milestone
  3. 转化为 **Executable Plan（exec plan）**

> 智能体应：
>
> - 主动指出需求歧义
> - 标记高风险模块
> - 给出工程级建议，而非 demo 方案

重点检查修正整个pipeline，从数据读取，正确处理做成dataset（不同type变量正确处理，新变量构建，剪裁等等），load进去训练，mlflow监控，evaluate, output，监控产物、结果检查，到pred等的全流程在本地跑通，并且每一步要合理，不能为了跑通做mock。。。整体上对标工程级项目，专业可靠好用可scaleable。请和我讨论理解需求便于指定计划

Now I have a thorough understanding of the entire codebase, the reference projects, and the ROOT file structure. Let me synthesize my analysis into a discussion.

------

## Analysis & Discussion

After thoroughly exploring BambooHepMl, weaver-core, Made-With-ML, and the actual ROOT file structure, here are the key findings and issues to discuss:

### Current State Summary

The framework already has a solid architecture: FeatureGraph DAG, DataConfig, model registry, Trainer, Evaluator, Predictor, MLflow tracker, ONNX export. However, **the existing pipeline has never been run against real ROOT data** — all integration tests use mock data with generic `met`/`Jet` structures that don't match the actual EDM4hep/podio ROOT files.

### Critical Gaps to Address

**1. Data Layer: ROOT branch name mapping is broken for your data**

Your ROOT files use EDM4hep podio naming like `EcalBarrelCollection.energy`, `MCParticle.momentum.x`, `TPCCollection.EDep`, etc. The current configs and tests assume collider physics naming (`Jet.pt`, `met`). The entire feature config needs to be rewritten to map the actual 23 features from the requirements to the real ROOT branch names.

Key question: the ROOT files have branches like `EcalBarrelCollection/EcalBarrelCollection.energy` — these are **variable-length arrays** (one energy per calorimeter hit per event). Your 23 features include both:

- **Aggregatable collections** (e.g., `EcalBarrelCollection.energy` — many hits → need sum/mean/max)
- **MCParticle fields** (e.g., `MCParticle.momentum.x` — also variable-length, but typically you want the primary particle, index 0?)

**Q1: For features like `EcalBarrelCollection.energy`, do you want `sum(energy)` as an event-level scalar, or keep it as a padded object-level array? The requirements say "all trimming (top-k / sum / mean / histogram) must be explicitly declared" — what strategy per detector subsystem?**

**2. Label Generation: Dictionary-format path → label**

For the 3-class classification task, the orchestrator already supports `label:path` format:

```yaml
source_path: "pi_3GeV:testdata/sim_pi-_3.00GeV*.root pi_5GeV:testdata/sim_pi-_5.00GeV*.root pi_7GeV:testdata/sim_pi-_7.00GeV*.root"
```

This will auto-generate `is_pi_3GeV`, `is_pi_5GeV`, `is_pi_7GeV` via `file_magic`. This mechanism exists but has never been tested against real files.

**3. Regression target: `_IncidentEnergy`**

This branch exists as a scalar float in `events;6` (1000 entries). It's straightforward for regression — map it as `complex` label type: `{_label_: _IncidentEnergy}`. But note: `events;5` (955 entries) doesn't have this branch, so treename must target `events;6` or handle missing gracefully.

**4. Feature engineering: all 23 features are variable-length**

Every single one of the 23 features is a sub-branch of a podio collection and is therefore a **variable-length array** (different number of hits/particles per event). There are no true scalars in the feature list. This means the pipeline needs a clear strategy for each:

| Category                  | Examples                                   | Likely strategy              |
| :------------------------ | :----------------------------------------- | :--------------------------- |
| Calorimeter energies (6)  | `EcalBarrelCollection.energy`              | `sum()` → event-level scalar |
| Contribution energies (2) | `EcalBarrelContributionCollection.energy`  | `sum()` → event-level scalar |
| Tracker EDep (6)          | `TPCCollection.EDep`, `VXDCollection.EDep` | `sum()` → event-level scalar |
| Lumical energy (1)        | `LumicalCollection.energy`                 | `sum()` → event-level scalar |
| Positions (10)            | `EcalBarrelCollection.position.x/y/z`      | `mean()` or weighted mean?   |
| MC momenta (5)            | `MCParticle.momentum.x/y/z`, `mass`        | Index-0 (primary particle)?  |

**Q2: Do you want all features reduced to event-level scalars (via sum/mean/first), or do you want some kept as padded object-level arrays for models that can handle sequences (like attention/transformer)?**

**5. Models: MLP vs FNN vs XGBoost**

Current code has `mlp_classifier` and `mlp_regressor`. The requirements mention "simple MLP, simple FNN, XGBoost." MLP and FNN are essentially the same (feed-forward network). XGBoost needs a separate code path — it doesn't use PyTorch, so it bypasses the Trainer entirely.

**Q3: For XGBoost, should it share the same FeatureGraph pipeline (features → numpy → xgboost), just skipping the torch DataLoader? This is the cleanest approach.**

**6. Output format**

Requirements say output should be a ROOT TTree with label, score branches, and observer variable branches. The current Predictor outputs Python dicts. We need a `write_root_output()` function using `uproot.WritableDirectory`.

**Q4: For classification output, you want branches like `label` (int), `score_pi_3GeV` (float), `score_pi_5GeV` (float), `score_pi_7GeV` (float), plus observer branches?**

**7. Evaluation**

The Evaluator exists but is minimal. Requirements ask for:

- Classification: ROC, confusion matrix, score distribution
- Regression: resolution, bias, pull distribution
- Auto-generated sanity check plots

Currently no plotting infrastructure exists.

**8. MLflow integration**

The tracker module exists but the actual `train_task` integration is shallow — it logs some basic metrics but doesn't log configs, artifacts, or model checkpoints to MLflow in a structured way. Needs tightening.

### Proposed Milestone Structure

1. **M0: Data Foundation** — Real ROOT reading, branch mapping, feature config for 23 features
2. **M1: Classification Pipeline** — 3-class with real data, training, evaluation plots, ROOT output
3. **M2: Regression Pipeline** — `_IncidentEnergy` target, resolution/bias/pull plots
4. **M3: XGBoost Integration** — Non-torch model path
5. **M4: MLflow & Reproducibility** — Full config/metric/artifact logging, seed control
6. **M5: Prediction Pipeline** — Standalone inference, ROOT output with scores + observers

Agent Reply — Clarified Decisions & Constraints (Authoritative)

Thanks for the detailed analysis — your understanding is largely correct. Below are authoritative decisions to remove ambiguity before drafting the execution plan.

The feature set now consists only of calorimeter / tracker hit–level collections.

Q1 & Q2 — Feature Reduction Strategy (Baseline, Mandatory)

All calorimeter and tracker hit–level collections are reduced to event-level scalars in the baseline pipeline.

No padded arrays, no sequence features, no attention models in Phase-1.

All reductions must be explicitly declared in FeatureConfig (no implicit behavior).

Reduction rules:

Energy-like quantities
(*.energy, *.EDep) → sum() per event

Position-like quantities
(*.position.x/y/z) → energy-weighted mean position per event
(i.e. Σ(x·E) / ΣE, similarly for y/z)

There are no true scalar branches in the input; all features originate from variable-length podio collections and must be reduced deterministically.

Sequence/object-level representations are explicitly deferred to a future phase and are out of scope for this execution plan.

Q3 — XGBoost Integration (Non-negotiable)

XGBoost must share the exact same FeatureGraph + DataConfig pipeline as Torch models.

Feature extraction produces numpy arrays once; model choice is downstream.

XGBoost bypasses the Torch Trainer/DataLoader, but does not get its own feature path.

This is mandatory to avoid feature drift and duplicated logic.

Q4 — Output ROOT Schema (Final)

Classification output:

## TTree: pred

event_id int
label int # argmax over scores
is_pi_3GeV bool # one-hot
is_pi_5GeV bool
is_pi_7GeV bool
score_pi_3GeV float # softmax probability
score_pi_5GeV float
score_pi_7GeV float
observer_* float # optional

Regression output:

## TTree: pred

event_id int
target_IncidentEnergy float
pred_IncidentEnergy float
residual float
pull float # if sigma is defined
observer_* float

Technical requirements:

ROOT writing via uproot.recreate() + WritableDirectory

Do not overwrite or modify the original events tree

Prediction output is written as a separate pred tree

Additional Global Constraints

Baseline pipeline must run end-to-end on real ROOT data, no mocks.

All feature transformations, configs, and artifacts must be MLflow-tracked.

Reproducibility (seed control, config logging) is mandatory.

With these constraints fixed, please proceed to draft the detailed executable plan aligned with the proposed milestones.Additional Constraint — Real ROOT First, Mock Is Disposable

One more important clarification based on the actual ROOT structure and target tasks:

The current framework contains extensive mock-based logic (mock Jet/MET features, synthetic datasets, placeholder tests/even config).

These mocks do not reflect the real EDM4hep / podio ROOT structure and must not constrain the design.

Directive:

The pipeline must be designed against the real ROOT files and real branches first.

Any existing mock code, configs, or tests that obstruct correct handling of:

podio-style variable-length collections

calorimeter / tracker hit aggregation

real feature reduction logic
may be boldly replaced or removed.

Mock components may be:

Deleted

Refactored to thin sanity tests



# **计划**

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



