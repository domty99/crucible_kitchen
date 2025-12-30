# Recipe Port Implementation Plan: DPO and Code RL

**Date:** 2025-12-30
**Status:** Implementation Plan
**Scope:** Port `preference/dpo` and `code_rl` recipes to tinkex_cookbook

---

## Executive Summary

This document details the complete implementation plan for porting two Python recipes (`preference/dpo` and `code_rl`) to the Elixir ecosystem. The implementation follows the "thin facade" pattern where:

- **tinkex_cookbook** remains thin (~200 LOC total) - config + adapter wiring only
- **crucible_kitchen** provides the orchestration stages (~600 LOC)
- **crucible_train** provides the core training logic (already exists)

---

## Audit Results: What Already Exists

### crucible_train (Fully Implemented)

| Module | Purpose | Status |
|--------|---------|--------|
| `Preference.TrainDPO` | DPO training loop | Exists (wraps supervised) |
| `Preference.DPODatasets` | Dataset building from comparisons | Exists |
| `Preference.Types` | Comparison, LabeledComparison | Exists |
| `RL.Train` | RL training loop | Exists |
| `RL.Rollouts` | `do_group_rollout/2` parallel collection | Exists |
| `RL.DataProcessing` | `assemble_training_data/2`, simple advantages | Exists |
| `RL.Metrics` | KL computation, `discounted_future_sum_vectorized/2` | Exists |
| `RL.Types` | Transition, Trajectory, TrajectoryGroup | Exists |
| `RL.Env`, `RL.EnvGroupBuilder` | Environment behaviours | Exists |
| `Distillation.TrainOnPolicy` | On-policy distillation | Exists |

### crucible_kitchen (Partial)

| Module | Purpose | Status |
|--------|---------|--------|
| `Workflows.Preference` | DPO workflow definition | Exists |
| `Workflows.Reinforcement` | RL workflow definition | Exists |
| `Workflows.Distillation` | Distillation workflow definition | Exists |
| `RLEnv.compute_gae/4` | Full GAE implementation | Exists (not integrated) |
| DPO Stages | 5 stages referenced by workflow | **MISSING** |
| RL Stages | 6 stages referenced by workflow | **MISSING** |
| Distillation Stages | 7 stages referenced by workflow | **MISSING** |

### tinkex_cookbook

| Component | Status |
|-----------|--------|
| Elixir recipes | **MISSING** |
| Mix tasks | `mix kitchen.run` exists |

---

## Repository Map

```
North-Shore-AI/
├── crucible_train/           # Core training logic (EXISTS)
│   └── lib/crucible_train/
│       ├── preference/       # DPO training
│       ├── rl/               # RL training
│       └── distillation/     # Distillation training
│
├── crucible_kitchen/         # Orchestration (PARTIAL)
│   └── lib/crucible_kitchen/
│       ├── workflows/        # Workflow definitions (EXISTS)
│       ├── stages/           # Stage implementations (MISSING 11)
│       └── adapters/         # Backend adapters (EXISTS)
│
├── tinkex_cookbook/          # Thin facade (MISSING)
│   └── lib/tinkex_cookbook/
│       ├── recipes/          # Recipe configs (TO CREATE)
│       └── adapters.ex       # Adapter presets (TO CREATE)
│
├── tinkex/                   # Tinker API client (EXISTS)
├── hf_datasets_ex/           # HuggingFace datasets (EXISTS)
└── crucible_harness/         # Experiment orchestration (EXISTS)
```

---

## Recipe 1: preference/dpo

### Python Source Analysis

```python
# tinker_cookbook/recipes/preference/dpo/train.py
@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "hhh"  # hhh, helpsteer3, ultrafeedback
    learning_rate: float = 1e-5
    dpo_beta: float = 0.1
    batch_size: int = 256
    reference_model_name: str | None = None
```

The Python recipe:
1. Loads preference dataset (HHH, HelpSteer3, UltraFeedback)
2. Builds DPO dataset from comparisons
3. Calls `train_dpo.main(config)` which delegates to supervised training

### Elixir Dependencies Used

| Dependency | Package | Usage |
|------------|---------|-------|
| DPO training types | `crucible_train` | `Preference.Comparison`, `Preference.LabeledComparison` |
| DPO dataset building | `crucible_train` | `Preference.DPODatasets.from_labeled/4` |
| Preference datasets | `hf_datasets_ex` | Load HHH, UltraFeedback, HelpSteer3 |
| Training client | `tinkex` | `Adapters.Tinkex.TrainingClient` |
| Telemetry | `crucible_kitchen` | Stage telemetry events |

### Missing Stages (crucible_kitchen)

#### 1. BuildPreferenceDataset (~60 LOC)

```elixir
# lib/crucible_kitchen/stages/build_preference_dataset.ex
defmodule CrucibleKitchen.Stages.BuildPreferenceDataset do
  @moduledoc """
  Builds preference dataset from raw comparison data.

  Wraps CrucibleTrain.Preference.DPODatasets.from_labeled/4
  """
  use CrucibleKitchen.Stage

  # Loads comparisons, converts to preference pairs
  # Sets :preference_dataset in context state
end
```

#### 2. GetPreferenceBatch (~40 LOC)

```elixir
# lib/crucible_kitchen/stages/get_preference_batch.ex
defmodule CrucibleKitchen.Stages.GetPreferenceBatch do
  @moduledoc """
  Gets next preference batch from dataset.
  """
  use CrucibleKitchen.Stage

  # Fetches batch with (prompt, chosen, rejected) triplets
  # Sets :preference_batch in context state
end
```

#### 3. ComputeReferenceLogprobs (~50 LOC)

```elixir
# lib/crucible_kitchen/stages/compute_reference_logprobs.ex
defmodule CrucibleKitchen.Stages.ComputeReferenceLogprobs do
  @moduledoc """
  Computes log probabilities from frozen reference model.

  Uses training_client to compute logprobs for chosen/rejected.
  """
  use CrucibleKitchen.Stage

  # Calls training_client.compute_logprobs/3 for ref model
  # Sets :ref_chosen_logprobs, :ref_rejected_logprobs in state
end
```

#### 4. DPOForwardBackward (~70 LOC)

```elixir
# lib/crucible_kitchen/stages/dpo_forward_backward.ex
defmodule CrucibleKitchen.Stages.DPOForwardBackward do
  @moduledoc """
  DPO forward-backward pass with beta-scaled preference loss.

  Loss = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected)))
  """
  use CrucibleKitchen.Stage

  # Calls training_client.dpo_forward_backward/3
  # Returns loss, gradients, metrics
end
```

#### 5. LogDPOMetrics (~40 LOC)

```elixir
# lib/crucible_kitchen/stages/log_dpo_metrics.ex
defmodule CrucibleKitchen.Stages.LogDPOMetrics do
  @moduledoc """
  Logs DPO-specific metrics: accuracy, margins, rewards.
  """
  use CrucibleKitchen.Stage

  # Emits [:crucible_kitchen, :dpo, :step] telemetry
  # Records accuracy, chosen_reward, rejected_reward, margin
end
```

### Recipe Config (tinkex_cookbook)

```elixir
# lib/tinkex_cookbook/recipes/dpo.ex (~80 LOC)
defmodule TinkexCookbook.Recipes.DPO do
  @moduledoc """
  Direct Preference Optimization recipe.
  """
  use CrucibleKitchen.Recipe

  def name, do: :dpo

  def default_config do
    %{
      model: "meta-llama/Llama-3.2-1B",
      dataset: :hhh,  # :hhh | :helpsteer3 | :ultrafeedback
      learning_rate: 1.0e-5,
      dpo_beta: 0.1,
      batch_size: 256,
      epochs: 1,
      reference_model: nil  # defaults to model
    }
  end

  def workflow, do: CrucibleKitchen.Workflows.Preference

  def required_adapters, do: [:training_client, :dataset_store]
end
```

### Data Flow

```
TinkexCookbook.Recipes.DPO (config)
         │
         ▼
CrucibleKitchen.Workflows.Preference
         │
         ├── LoadDataset ─► hf_datasets_ex (HHH/UltraFeedback)
         ├── InitSession ─► tinkex TrainingClient
         ├── InitTokenizer ─► tinkex
         ├── BuildPreferenceDataset ─► crucible_train DPODatasets
         │
         └── Loop (epochs)
              └── Loop (batches)
                   ├── GetPreferenceBatch ─► dataset iteration
                   ├── ComputeReferenceLogprobs ─► training_client
                   ├── DPOForwardBackward ─► training_client
                   ├── AwaitFuture
                   ├── OptimStep ─► training_client
                   ├── AwaitFuture
                   └── LogDPOMetrics ─► telemetry
```

---

## Recipe 2: code_rl

### Python Source Analysis

```python
# tinker_cookbook/recipes/code_rl/train.py
@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
```

The Python recipe:
1. Creates DeepcoderDatasetBuilder (code generation environment)
2. Collects rollouts via sampling
3. Computes advantages and trains with RL

### Elixir Dependencies Used

| Dependency | Package | Usage |
|------------|---------|-------|
| RL types | `crucible_train` | `RL.Trajectory`, `RL.TrajectoryGroup` |
| Rollout collection | `crucible_train` | `RL.Rollouts.do_group_rollout/2` |
| Data processing | `crucible_train` | `RL.DataProcessing.assemble_training_data/2` |
| GAE computation | `crucible_kitchen` | `RLEnv.compute_gae/4` |
| Nx tensors | `Nx` | Discounted rewards, advantage normalization |
| Training client | `tinkex` | `Adapters.Tinkex.TrainingClient` |
| Experiment harness | `crucible_harness` | Evaluation orchestration |

### Missing Stages (crucible_kitchen)

#### 1. BuildEnvGroup (~60 LOC)

```elixir
# lib/crucible_kitchen/stages/build_env_group.ex
defmodule CrucibleKitchen.Stages.BuildEnvGroup do
  @moduledoc """
  Builds environment group for rollout collection.

  Wraps CrucibleTrain.RL.EnvGroupBuilder
  """
  use CrucibleKitchen.Stage

  # Creates env_group from config
  # Sets :env_group in context state
end
```

#### 2. DoRollout (~80 LOC)

```elixir
# lib/crucible_kitchen/stages/do_rollout.ex
defmodule CrucibleKitchen.Stages.DoRollout do
  @moduledoc """
  Collects trajectories via parallel rollouts.

  Wraps CrucibleTrain.RL.Rollouts.do_group_rollout/2
  """
  use CrucibleKitchen.Stage

  # Collects TrajectoryGroup with rewards
  # Sets :trajectory_group in context state
end
```

#### 3. ComputeAdvantages (~70 LOC)

```elixir
# lib/crucible_kitchen/stages/compute_advantages.ex
defmodule CrucibleKitchen.Stages.ComputeAdvantages do
  @moduledoc """
  Computes Generalized Advantage Estimation (GAE).

  Uses CrucibleKitchen.RLEnv.compute_gae/4
  """
  use CrucibleKitchen.Stage

  # Computes GAE with gamma=0.99, lambda=0.95
  # Sets :advantages, :returns in context state
end
```

#### 4. AssembleRLBatch (~50 LOC)

```elixir
# lib/crucible_kitchen/stages/assemble_rl_batch.ex
defmodule CrucibleKitchen.Stages.AssembleRLBatch do
  @moduledoc """
  Assembles trajectory data into training batch.

  Wraps CrucibleTrain.RL.DataProcessing.assemble_training_data/2
  """
  use CrucibleKitchen.Stage

  # Converts trajectories to training batch
  # Sets :rl_batch in context state
end
```

#### 5. PPOUpdate (~80 LOC)

```elixir
# lib/crucible_kitchen/stages/ppo_update.ex
defmodule CrucibleKitchen.Stages.PPOUpdate do
  @moduledoc """
  PPO policy gradient update with clipping.

  Implements clipped surrogate objective:
  L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
  """
  use CrucibleKitchen.Stage

  # Calls training_client.ppo_step/3
  # Returns policy_loss, value_loss, entropy
end
```

#### 6. LogRLMetrics (~50 LOC)

```elixir
# lib/crucible_kitchen/stages/log_rl_metrics.ex
defmodule CrucibleKitchen.Stages.LogRLMetrics do
  @moduledoc """
  Logs RL-specific metrics: rewards, KL, entropy.
  """
  use CrucibleKitchen.Stage

  # Emits [:crucible_kitchen, :rl, :step] telemetry
  # Records reward_mean, reward_std, kl_div, entropy, clip_fraction
end
```

### Recipe Config (tinkex_cookbook)

```elixir
# lib/tinkex_cookbook/recipes/code_rl.ex (~80 LOC)
defmodule TinkexCookbook.Recipes.CodeRL do
  @moduledoc """
  Code generation via Reinforcement Learning recipe.
  """
  use CrucibleKitchen.Recipe

  def name, do: :code_rl

  def default_config do
    %{
      model: "meta-llama/Llama-3.1-8B-Instruct",
      env: :deepcoder,
      group_size: 4,
      groups_per_batch: 100,
      learning_rate: 1.0e-5,
      gamma: 0.99,
      gae_lambda: 0.95,
      clip_epsilon: 0.2,
      ppo_epochs: 4,
      kl_penalty_coef: 0.0,
      num_rollouts: 100
    }
  end

  def workflow, do: CrucibleKitchen.Workflows.Reinforcement

  def required_adapters, do: [:training_client, :dataset_store]
end
```

### Environment Adapter (tinkex_cookbook)

```elixir
# lib/tinkex_cookbook/envs/deepcoder.ex (~80 LOC)
defmodule TinkexCookbook.Envs.Deepcoder do
  @moduledoc """
  Deepcoder code generation environment.

  Implements CrucibleTrain.RL.Env behaviour.
  """
  @behaviour CrucibleTrain.RL.Env

  # Problems from Deepcoder dataset
  # Reward: 1.0 if code executes correctly, 0.0 otherwise
end
```

### Data Flow

```
TinkexCookbook.Recipes.CodeRL (config)
         │
         ▼
CrucibleKitchen.Workflows.Reinforcement
         │
         ├── LoadDataset ─► hf_datasets_ex (Deepcoder problems)
         ├── InitSession ─► tinkex TrainingClient
         ├── InitTokenizer ─► tinkex
         ├── BuildEnvGroup ─► TinkexCookbook.Envs.Deepcoder
         │
         └── Loop (rollouts)
              ├── DoRollout ─► crucible_train RL.Rollouts
              ├── ComputeAdvantages ─► CrucibleKitchen.RLEnv.compute_gae
              ├── AssembleRLBatch ─► crucible_train RL.DataProcessing
              ├── PPOUpdate ─► training_client
              ├── LogRLMetrics ─► telemetry
              │
              ├── Conditional (checkpoint)
              └── Conditional (evaluate) ─► crucible_harness
```

---

## Implementation Summary

### Code Distribution

| Location | Files | LOC | Purpose |
|----------|-------|-----|---------|
| crucible_kitchen/stages/ | 11 new | ~600 | Stage implementations |
| tinkex_cookbook/recipes/ | 2 new | ~160 | Recipe configs |
| tinkex_cookbook/envs/ | 1 new | ~80 | Deepcoder env |
| tinkex_cookbook/adapters.ex | 1 new | ~40 | Adapter presets |
| Tests | ~10 new | ~400 | Test coverage |
| **Total** | **~25** | **~1280** | |

### Files to Create

```
crucible_kitchen/lib/crucible_kitchen/stages/
├── build_preference_dataset.ex     # DPO
├── get_preference_batch.ex         # DPO
├── compute_reference_logprobs.ex   # DPO
├── dpo_forward_backward.ex         # DPO
├── log_dpo_metrics.ex              # DPO
├── build_env_group.ex              # RL
├── do_rollout.ex                   # RL
├── compute_advantages.ex           # RL
├── assemble_rl_batch.ex            # RL
├── ppo_update.ex                   # RL
└── log_rl_metrics.ex               # RL

tinkex_cookbook/lib/tinkex_cookbook/
├── recipes/
│   ├── dpo.ex
│   └── code_rl.ex
├── envs/
│   └── deepcoder.ex
└── adapters.ex
```

### Dependency Graph

```
tinkex_cookbook
    │
    ├──► crucible_kitchen (workflows, stages)
    │         │
    │         ├──► crucible_train (training logic)
    │         │         │
    │         │         └──► tinkex (Tinker API)
    │         │
    │         └──► Nx (tensor ops)
    │
    ├──► hf_datasets_ex (datasets)
    │
    └──► crucible_harness (evaluation)
```

---

## Test Plan

### Unit Tests

1. **DPO Stages** - Test each stage in isolation with noop adapters
2. **RL Stages** - Test each stage in isolation with noop adapters
3. **Recipes** - Test config validation, default values

### Integration Tests

1. **DPO Workflow** - End-to-end with noop adapters
2. **RL Workflow** - End-to-end with noop adapters
3. **Telemetry** - Verify all events emitted

### Smoke Tests

1. **DPO with Tinker** - Real training with small dataset
2. **Code RL with Tinker** - Real training with Deepcoder

---

## Implementation Order

1. **Phase 1: DPO Stages** (crucible_kitchen)
   - BuildPreferenceDataset
   - GetPreferenceBatch
   - ComputeReferenceLogprobs
   - DPOForwardBackward
   - LogDPOMetrics

2. **Phase 2: DPO Recipe** (tinkex_cookbook)
   - DPO recipe config
   - Adapters preset

3. **Phase 3: RL Stages** (crucible_kitchen)
   - BuildEnvGroup
   - DoRollout
   - ComputeAdvantages
   - AssembleRLBatch
   - PPOUpdate
   - LogRLMetrics

4. **Phase 4: RL Recipe** (tinkex_cookbook)
   - CodeRL recipe config
   - Deepcoder environment

5. **Phase 5: Tests**
   - Unit tests
   - Integration tests

6. **Phase 6: Documentation**
   - Update READMEs
   - Usage examples
