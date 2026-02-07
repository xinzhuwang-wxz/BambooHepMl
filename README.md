# BambooHepMl

A machine learning framework for high energy physics, combining
DAG-based feature engineering with a complete ML pipeline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

BambooHepMl provides a configuration-driven workflow for training,
evaluating, and serving ML models on HEP detector data. It reads
variable-length collections from EDM4hep/podio ROOT files, reduces them
to event-level scalars through a declarative feature graph, and feeds
the result into PyTorch or XGBoost models. The full pipeline -- from raw
ROOT files to ONNX export and FastAPI serving -- is controlled by three
YAML configuration files.

## Architecture

```
bamboohepml/
  cli.py                 # Typer CLI entry point
  config.py              # Global logging and MLflow setup
  metadata.py            # Model metadata persistence
  utils.py               # Shared utilities
  data/
    dataset.py           # HEPDataset (PyTorch Dataset)
    config.py            # DataConfig (labels, branches, splits)
    preprocess.py        # Normalization and preprocessing
    fileio.py            # ROOT/Parquet/HDF5 I/O helpers
    features/
      feature_graph.py   # DAG-based feature computation
      expression.py      # Expression engine (safe_sum, etc.)
      processors.py      # Normalizer, Clipper, Padder
    sources/
      factory.py         # Auto-detect file format
      root_source.py     # uproot-based ROOT reader
      parquet_source.py  # Parquet reader
      hdf5_source.py     # HDF5 reader
  models/
    base.py              # BaseModel, ClassificationModel, RegressionModel
    registry.py          # @register_model decorator and get_model()
    common/mlp.py        # MLPClassifier, MLPRegressor
  engine/
    predictor.py         # Batch prediction loop
  pipeline/
    orchestrator.py      # PipelineOrchestrator (unified entry point)
    state.py             # PipelineState serialization
  tasks/
    train.py             # Training task (Torch + XGBoost)
    predict.py           # Prediction task
    export.py            # ONNX export
    inspect.py           # Data and feature inspection
  scheduler/
    local.py             # LocalScheduler
    slurm.py             # SLURMScheduler
  serve/
    cli.py               # serve fastapi / serve ray sub-commands
    fastapi_server.py    # FastAPI inference server
    ray_serve.py         # Ray Serve deployment
    onnx_predictor.py    # ONNX Runtime predictor
  experiment/
    tracker.py           # Experiment tracker interface
    mlflow_tracker.py    # MLflow integration
    tensorboard_tracker.py
configs/                 # Example YAML configurations
tests/                   # Test suite (pytest)
```

## Getting started

### Prerequisites

- Python 3.9 or later
- PyTorch 1.12 or later
- uproot 5, awkward 2 (for ROOT file I/O)

### Installation

```bash
git clone https://github.com/xinzhuwang-wxz/BambooHepMl.git
cd BambooHepMl
pip install -e .
```

To install test and serve extras:

```bash
pip install -e ".[test]"
pip install -e ".[serve]"
```

## Quick start

After installation, a minimal end-to-end run requires three steps:

1.  Prepare your configuration files (or use the provided examples):

    ```
    configs/
      pipeline_edm4hep.yaml          # paths, model, training params
      features_edm4hep.yaml          # feature definitions
      data_edm4hep_classification.yaml   # label scheme
    ```

2.  Train a classification model:

    ```bash
    bamboohepml train -c configs/pipeline_edm4hep.yaml
    ```

    This reads the ROOT files specified in `data.source_path`, builds the
    feature graph, trains an MLP classifier, and saves the best model to
    `outputs/edm4hep/`.

3.  Run prediction on the test split:

    ```bash
    bamboohepml predict -c configs/pipeline_edm4hep.yaml \
      -m outputs/edm4hep/best_model.pt -o predictions.root
    ```

To switch to a regression task, point the pipeline at the regression
data config (`data_edm4hep_regression.yaml`) and the framework
automatically selects `MLPRegressor` and MSE loss.

## Usage

BambooHepMl provides five CLI commands.

```bash
# Train a model
bamboohepml train -c configs/pipeline_edm4hep.yaml

# Inspect data and features before training
bamboohepml inspect -c configs/pipeline_edm4hep.yaml

# Run prediction with a trained model
bamboohepml predict -c configs/pipeline_edm4hep.yaml \
  -m outputs/model.pt -o predictions.root

# Export to ONNX
bamboohepml export -m outputs/model.pt -o model.onnx

# Start a FastAPI inference server
bamboohepml serve fastapi --model-path outputs/model.pt \
  --metadata-path outputs/metadata.json
```

All commands accept `--scheduler slurm --slurm-config slurm.sh` for
batch submission.

## Configuration

The framework is driven by three YAML files.

### Pipeline configuration

`pipeline_edm4hep.yaml` defines data paths, model architecture, training
hyper-parameters, and output settings:

```yaml
data:
  source_path: "data/sim_*.root"
  features_config: "configs/features_edm4hep.yaml"
  train_range: [0.0, 0.7]
  val_range: [0.7, 0.85]
  test_range: [0.85, 1.0]

model:
  hidden_dims: [128, 64, 32]
  dropout: 0.1
  activation: relu
  batch_norm: true
  embed_dim: 64

training:
  num_epochs: 5
  batch_size: 128
  learning_rate: 0.001
  optimizer: adam
  early_stopping:
    patience: 10
    monitor: val_loss

output:
  base_dir: "outputs/edm4hep"
  save_best: true
  export_root: true

mlflow:
  enabled: true
```

### Feature configuration

`features_edm4hep.yaml` declares every feature used in the pipeline.
Each feature specifies a full ROOT path in the `source` field and a
`reduction` block for variable-length collections. No alias tables or
`expr` strings are needed.

```yaml
features:
  event_level:
    - name: ecal_barrel_energy
      source: "EcalBarrelCollection/EcalBarrelCollection.energy"
      type: event
      reduction:
        type: safe_sum
      normalize:
        method: auto

    - name: ecal_barrel_pos_x
      source: "EcalBarrelCollection/EcalBarrelCollection.position.x"
      type: event
      reduction:
        type: energy_weighted_mean
        weight: "EcalBarrelCollection/EcalBarrelCollection.energy"
      normalize:
        method: auto
```

Supported reduction types:

| Reduction | Description |
|-----------|-------------|
| `safe_sum` | Sum over variable-length collection per event |
| `energy_weighted_mean` | Weighted mean using an energy branch as weight |

### Data configuration

Data YAML files define the ROOT tree name, label scheme, and optional
file-magic rules for automatic label assignment from filenames.

Classification example (`data_edm4hep_classification.yaml`):

```yaml
treename: "events"

labels:
  type: simple
  value:
    - is_3GeV
    - is_5GeV
    - is_7GeV

file_magic:
  is_3GeV:
    "3\\.00GeV": 1
  is_5GeV:
    "5\\.00GeV": 1
  is_7GeV:
    "7\\.00GeV": 1
```

Regression example (`data_edm4hep_regression.yaml`):

```yaml
treename: "events"

labels:
  type: complex
  value:
    _label_: "_IncidentEnergy"
```

For regression, the `complex` label type maps `_label_` to a ROOT branch
name whose values become continuous targets.

## Data sources

The `DataSourceFactory` automatically selects a reader based on file
extension:

| Extension | Reader |
|-----------|--------|
| `.root` | `ROOTDataSource` (uproot) |
| `.parquet` | `ParquetDataSource` |
| `.h5`, `.hdf5` | `HDF5DataSource` |

Glob patterns are supported in `source_path` (e.g. `data/sim_*.root`).

## Models

Models are registered via the `@register_model` decorator and created at
runtime through `get_model()`.

Built-in models:

| Name | Class | Task |
|------|-------|------|
| `mlp_classifier` | `MLPClassifier` | Classification |
| `mlp_regressor` | `MLPRegressor` | Regression |

Both accept `event_input_dim` and `object_input_dim` parameters. The
`PipelineOrchestrator` infers these dimensions automatically from the
feature graph. The forward method takes a batch dict with `"event"` and
optionally `"object"` and `"mask"` keys.

## Development

### Code style

The project uses black, isort, and flake8 with a 150-character line
limit. CI enforces these on every push.

```bash
black bamboohepml tests
isort bamboohepml tests
flake8 bamboohepml
```

### Running tests

```bash
pytest tests/ -v --tb=short
```

### Documentation

```bash
pip install mkdocs "mkdocstrings[python]"
mkdocs serve     # local preview
mkdocs build     # static build
```

## License

[MIT](https://opensource.org/licenses/MIT)

## Acknowledgments

BambooHepMl draws inspiration from
[weaver-core](https://github.com/colizz/weaver-core) and
[Made-With-ML](https://github.com/GokuMohandas/Made-With-ML).
