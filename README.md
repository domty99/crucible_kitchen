<p align="center">
  <img src="assets/crucible_kitchen.svg" alt="Crucible Kitchen" width="200" />
</p>

<h1 align="center">Crucible Kitchen</h1>

<p align="center">
  <strong>Industrial ML training orchestration — backend-agnostic workflow engine for supervised, reinforcement, and preference learning.</strong>
</p>

<p align="center">
  <a href="https://hex.pm/packages/crucible_kitchen"><img src="https://img.shields.io/hexpm/v/crucible_kitchen.svg?style=flat-square&color=6e4a7e" alt="Hex Version" /></a>
  <a href="https://hexdocs.pm/crucible_kitchen"><img src="https://img.shields.io/badge/docs-hexdocs-5e60ce?style=flat-square" alt="Docs" /></a>
  <a href="https://github.com/North-Shore-AI/crucible_kitchen/actions"><img src="https://img.shields.io/github/actions/workflow/status/North-Shore-AI/crucible_kitchen/ci.yml?style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License" /></a>
</p>

---

## Overview

Crucible Kitchen is the missing orchestration layer that makes ML cookbooks trivially thin. It provides:

- **Pre-built workflows** for supervised, RL, DPO, and distillation training
- **Declarative workflow DSL** for composing custom pipelines
- **Port/adapter pattern** for backend flexibility (Tinker, Fireworks, Modal, local Nx)
- **Model evaluation** with metrics (accuracy, F1, precision, recall)
- **Model registration** with lineage tracking and artifact versioning
- **Comprehensive telemetry** for observability
- **First-class reproducibility** with deterministic seeding and artifact versioning

```
┌─────────────────────────────────────────────────────────────────┐
│                    COOKBOOK FRONTENDS                           │
│  tinkex_cookbook    fireworks_cookbook    modal_cookbook        │
│  (config + adapters only, <2K LOC each)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CRUCIBLE KITCHEN                            │
│  Recipes → Workflows → Stages → Ports                           │
│  (backend-agnostic orchestration)                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Training   │  │  Evaluation  │  │   Registry   │          │
│  │   Stages     │  │   Metrics    │  │   Lineage    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND ADAPTERS                             │
│  Tinker, Fireworks, Modal, LocalNx, Noop (testing)              │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

Add `crucible_kitchen` to your dependencies:

```elixir
def deps do
  [
    {:crucible_kitchen, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# 1. Define adapters for your backend
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []},
  evaluator: {CrucibleKitchen.Adapters.Noop.Evaluator, []},
  model_registry: {CrucibleKitchen.Adapters.Noop.ModelRegistry, []}
}

# 2. Configure your training run
config = %{
  model: "meta-llama/Llama-3.1-8B",
  dataset: "HuggingFaceH4/no_robots",
  epochs: 3,
  batch_size: 128,
  learning_rate: 2.0e-4
}

# 3. Run!
{:ok, result} = CrucibleKitchen.run(:supervised, config, adapters: adapters)

# 4. Access results
result.context.state.eval_results     # => %{accuracy: 0.92, f1: 0.89, ...}
result.context.state.registered_model # => %{id: "...", name: "...", version: "1.0.0"}
```

## DPO Training (Preference Learning)

```elixir
# Direct Preference Optimization for alignment
config = %{
  model: "meta-llama/Llama-3.2-1B",
  dataset: :hhh,  # or :ultrafeedback, :helpsteer3
  dpo_beta: 0.1,
  epochs: 1,
  batch_size: 256,
  learning_rate: 1.0e-5
}

{:ok, result} = CrucibleKitchen.run(:preference, config, adapters: adapters)

# Access DPO metrics
result.context.state.dpo_metrics
# => %{loss: 0.32, accuracy: 0.78, margin: 1.2, chosen_reward: 0.8, rejected_reward: -0.4}
```

## RL Training (Reinforcement Learning)

```elixir
# GRPO-style RL training for code generation
config = %{
  model: "meta-llama/Llama-3.1-8B-Instruct",
  env: :deepcoder,
  group_size: 4,
  groups_per_batch: 100,
  num_rollouts: 100,
  gamma: 0.99,
  gae_lambda: 0.95,
  clip_epsilon: 0.2,
  ppo_epochs: 4,
  learning_rate: 1.0e-5
}

{:ok, result} = CrucibleKitchen.run(:reinforcement, config, adapters: adapters)

# Access RL metrics
result.context.state.rollout_metrics
# => %{reward_mean: 0.75, reward_std: 0.15, num_trajectories: 400}

result.context.state.ppo_metrics
# => %{policy_loss: 0.02, entropy: 1.2, clip_fraction: 0.12}
```

## Supervised Training Pipeline

The `:supervised` workflow executes the following stages:

```
load_dataset → init_session → training_loop → save_final → evaluate → register_model → cleanup
                                   │
                          ┌────────┴────────┐
                          │  For each epoch │
                          │    └─ batches   │
                          │       └─ step   │
                          └─────────────────┘
```

1. **Load Dataset** — Fetches and prepares training data
2. **Init Session** — Starts training session with backend
3. **Training Loop** — Epochs → Batches → Forward/Backward + Optimizer
4. **Save Final** — Persists final model weights
5. **Evaluate** — Computes accuracy, F1, precision, recall
6. **Register Model** — Registers in model registry with lineage
7. **Cleanup** — Releases resources

## Key Concepts

### Recipes

Configuration-driven training definitions. Recipes declare WHAT to train, not HOW.

```elixir
defmodule MyCookbook.Recipes.SlBasic do
  use CrucibleKitchen.Recipe

  def name, do: :sl_basic
  def description, do: "Basic supervised fine-tuning"
  def workflow, do: CrucibleKitchen.Workflows.Supervised.__workflow__()

  def default_config do
    %{
      model: "meta-llama/Llama-3.1-8B",
      epochs: 1,
      batch_size: 128
    }
  end

  def required_adapters, do: [:training_client, :dataset_store]
  def optional_adapters, do: [:evaluator, :model_registry, :blob_store]
end
```

### Workflows

Compositions of stages with control flow. Workflows define HOW training is orchestrated.

```elixir
defmodule MyWorkflow do
  use CrucibleKitchen.Workflow

  workflow do
    stage :load, LoadDatasetStage
    stage :init, InitSessionStage

    loop :epochs, over: fn ctx -> 0..(ctx.config.epochs - 1) end do
      loop :batches, over: fn ctx -> ctx.state.dataset end do
        stage :forward_backward, ForwardBackwardStage
        stage :optim_step, OptimStepStage
      end

      conditional fn ctx -> should_checkpoint?(ctx) end do
        stage :checkpoint, CheckpointStage
      end
    end

    stage :save, SaveWeightsStage
    stage :evaluate, EvaluateStage
    stage :register, RegisterModelStage
  end
end
```

### Stages

Atomic units of work with lifecycle hooks.

```elixir
defmodule ForwardBackwardStage do
  use CrucibleKitchen.Stage

  @impl true
  def name, do: :forward_backward

  @impl true
  def execute(context) do
    {adapter, opts} = Context.get_adapter(context, :training_client)
    session = Context.get_state(context, :session)
    batch = Context.get_state(context, :current_batch)

    {:ok, result} = adapter.forward_backward(opts, session, batch)
    {:ok, Context.put_state(context, :fb_result, result)}
  end
end
```

### Ports & Adapters

Behaviour contracts for external integrations.

| Port | Purpose | Required |
|------|---------|----------|
| `TrainingClient` | Backend training operations (SFT, DPO, RL) | Yes |
| `DatasetStore` | Dataset loading and streaming | Yes |
| `Evaluator` | Model evaluation metrics | No |
| `ModelRegistry` | Model versioning and lineage | No |
| `BlobStore` | Artifact storage (checkpoints) | No |
| `HubClient` | Model hub operations | No |
| `FeedbackClient` | Production feedback and drift detection | No |

```elixir
# Port definition
defmodule CrucibleKitchen.Ports.Evaluator do
  @callback evaluate(opts(), model(), dataset()) :: {:ok, results()} | {:error, term()}
  @callback generate_report(opts(), results()) :: {:ok, report()} | {:error, term()}
end

# Adapter implementation
defmodule CrucibleKitchen.Adapters.Evaluator.EvalClient do
  @behaviour CrucibleKitchen.Ports.Evaluator

  @impl true
  def evaluate(opts, model, dataset) do
    # Uses EvalEx for metric computation
    EvalEx.Harness.run(model, dataset, opts)
  end
end
```

## Built-in Workflows

| Workflow | Description | Status |
|----------|-------------|--------|
| `:supervised` | Standard supervised fine-tuning (SFT) with eval & registry | ✅ Implemented |
| `:reinforcement` | RL with rollouts, GAE, and PPO (GRPO-style) | ✅ Implemented |
| `:preference` | Direct Preference Optimization (DPO) | ✅ Implemented |
| `:distillation` | Knowledge distillation | 🚧 Planned |
| `:feedback_loop` | Production feedback-driven retraining | ✅ Implemented |

### Reinforcement Learning Pipeline

The `:reinforcement` workflow implements GRPO-style RL training:

```
build_env_group → rollouts_loop → save_final → cleanup
                        │
               ┌────────┴────────┐
               │  For each batch │
               │    └─ rollout   │
               │    └─ advantages│
               │    └─ ppo_update│
               └─────────────────┘
```

**Stages:**
- `BuildEnvGroup` — Creates parallel environment group for rollouts
- `DoRollout` — Collects trajectories via model sampling
- `ComputeAdvantages` — Generalized Advantage Estimation (GAE)
- `AssembleRLBatch` — Assembles trajectories into training batch
- `PPOUpdate` — PPO policy gradient with clipping
- `LogRLMetrics` — Logs reward, policy loss, entropy, clip fraction

### Preference Learning Pipeline (DPO)

The `:preference` workflow implements Direct Preference Optimization:

```
load_dataset → init_session → build_prefs → training_loop → save_final → cleanup
                                                  │
                                         ┌────────┴────────┐
                                         │  For each epoch │
                                         │    └─ batches   │
                                         │       └─ dpo    │
                                         └─────────────────┘
```

**Stages:**
- `BuildPreferenceDataset` — Builds preference pairs from raw comparisons
- `GetPreferenceBatch` — Gets next batch of (prompt, chosen, rejected) tuples
- `ComputeReferenceLogprobs` — Computes frozen reference model logprobs
- `DPOForwardBackward` — DPO loss with beta-scaled preference margin
- `LogDPOMetrics` — Logs loss, accuracy, chosen/rejected rewards, margin

### Feedback Loop Pipeline

The `:feedback_loop` workflow enables production-driven retraining:

```
check_triggers → curate_data → export_data → update_baseline
```

**Stages:**
- `CheckTriggers` — Monitors drift, quality degradation, data volume
- `CurateData` — Selects high-value examples via active learning
- `ExportFeedbackData` — Exports curated data for training
- `UpdateBaseline` — Updates drift detection baseline post-training

## Telemetry

All operations emit telemetry events for observability:

```elixir
# Attach console handler
CrucibleKitchen.Telemetry.attach_console_handler()

# Or attach all default handlers
CrucibleKitchen.Telemetry.attach_default_handlers()
```

### Events

#### Core Events

| Event | Measurements | Metadata |
|-------|--------------|----------|
| `[:crucible_kitchen, :workflow, :start]` | - | recipe, config |
| `[:crucible_kitchen, :workflow, :stop]` | duration | recipe, result |
| `[:crucible_kitchen, :stage, :start]` | - | stage, recipe |
| `[:crucible_kitchen, :stage, :stop]` | duration | stage, recipe |
| `[:crucible_kitchen, :training, :step]` | loss, lr, tokens_per_sec | step, epoch |
| `[:crucible_kitchen, :training, :epoch]` | avg_loss | epoch, steps |
| `[:crucible_kitchen, :training, :checkpoint]` | - | step, path |
| `[:crucible_kitchen, :eval, :complete]` | accuracy, f1, precision, recall | step, sample_count |
| `[:crucible_kitchen, :model, :registered]` | - | model_id, version, name |

#### DPO Events

| Event | Measurements | Metadata |
|-------|--------------|----------|
| `[:crucible_kitchen, :dpo, :step]` | loss, accuracy, chosen_reward, rejected_reward, margin | step, beta, num_pairs |
| `[:crucible_kitchen, :dpo, :dataset_built]` | num_pairs, num_batches | batch_size |

#### RL Events

| Event | Measurements | Metadata |
|-------|--------------|----------|
| `[:crucible_kitchen, :rl, :env_group_built]` | group_size, groups_per_batch | env_type |
| `[:crucible_kitchen, :rl, :rollout_complete]` | reward_mean, reward_std, num_trajectories | rollout_index |
| `[:crucible_kitchen, :rl, :advantages_computed]` | advantage_mean, advantage_std, return_mean, return_std, num_steps | - |
| `[:crucible_kitchen, :rl, :ppo_update]` | policy_loss, value_loss, entropy, clip_fraction, kl_divergence | epoch, batch_size |
| `[:crucible_kitchen, :rl, :step]` | reward_mean, policy_loss, entropy, clip_fraction | step |

#### Feedback Loop Events

| Event | Measurements | Metadata |
|-------|--------------|----------|
| `[:crucible_kitchen, :feedback, :triggers_checked]` | drift_score, quality_score, data_volume | deployment_id |
| `[:crucible_kitchen, :feedback, :data_curated]` | num_selected, total_available | strategy |
| `[:crucible_kitchen, :feedback, :data_exported]` | num_examples, export_path | format |

## Testing

Use noop adapters for testing without a real backend:

```elixir
# All noop adapters
{:ok, result} = CrucibleKitchen.run(:supervised, config,
  adapters: %{
    training_client: {CrucibleKitchen.Adapters.Noop.TrainingClient, []},
    dataset_store: {CrucibleKitchen.Adapters.Noop.DatasetStore, []},
    evaluator: {CrucibleKitchen.Adapters.Noop.Evaluator, []},
    model_registry: {CrucibleKitchen.Adapters.Noop.ModelRegistry, []}
  })

# Dry run (validation only)
{:ok, %{validated: true}} = CrucibleKitchen.run(:supervised, config,
  adapters: adapters,
  dry_run: true)
```

## Documentation

- [Executive Summary](docs/00_EXECUTIVE_SUMMARY.md)
- [Architecture Patterns](docs/01_ARCHITECTURE_PATTERNS.md)
- [Component Design](docs/02_COMPONENT_DESIGN.md)
- [Workflow Engine](docs/03_WORKFLOW_ENGINE.md)
- [API Surface](docs/04_API_SURFACE.md)
- [Implementation Roadmap](docs/05_IMPLEMENTATION_ROADMAP.md)
- [Ecosystem Integration](docs/06_ECOSYSTEM_INTEGRATION.md)

## Part of the Crucible Ecosystem

Crucible Kitchen integrates with:

| Project | Purpose |
|---------|---------|
| `crucible_train` | Types, renderers, training primitives |
| `crucible_ir` | Experiment specifications |
| `crucible_framework` | Pipeline orchestration |
| `crucible_telemetry` | Research-grade instrumentation |
| `crucible_harness` | Batch experiment management |
| `crucible_model_registry` | Model versioning and lineage |
| `eval_ex` | Model evaluation harness |

## License

MIT License
