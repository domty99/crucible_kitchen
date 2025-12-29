# tinker-cookbook to crucible_ Ecosystem Integration Mapping

**Date**: 2025-12-28
**Status**: Master Planning Document
**Purpose**: Complete mapping for porting tinker-cookbook functionality to the Elixir ecosystem

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Ecosystem Overview](#2-ecosystem-overview)
3. [Module-by-Module Mapping](#3-module-by-module-mapping)
4. [Dependency Diagram](#4-dependency-diagram)
5. [Gap Analysis](#5-gap-analysis)
6. [New Projects Needed](#6-new-projects-needed)
7. [Interface Design](#7-interface-design)
8. [Implementation Phases](#8-implementation-phases)

---

## 1. Executive Summary

### Architecture Vision

```
tinkex_cookbook (thin config layer, <500 LOC)
         |
         v
crucible_kitchen (orchestration, workflows, stages)
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
crucible_   tinkex   hf_    crucible_   crucible_
train    (training)  hub_ex  datasets   model_registry
```

### Key Insight

tinker-cookbook is ~10K LOC Python implementing:
- Training loops (SFT, RL, DPO, distillation)
- Renderers/tokenization
- Dataset handling
- Logging/metrics
- Utility functions

The crucible_ ecosystem already has ~80% of this functionality. `tinkex_cookbook` should be a **thin configuration layer** (<500 LOC) that:
1. Provides recipes and config only
2. References adapters implemented in `crucible_kitchen`
3. Contains NO training logic (that lives in `crucible_kitchen` and `crucible_train`)

---

## 2. Ecosystem Overview

### Existing Projects

| Project | Role | LOC | Completeness |
|---------|------|-----|--------------|
| `crucible_kitchen` | Orchestration layer | ~2K | 60% |
| `crucible_train` | Training infrastructure | ~5K | 80% |
| `crucible_datasets` | Dataset management | ~2K | 70% |
| `crucible_model_registry` | Model versioning | ~1.5K | 90% |
| `hf_hub_ex` | HuggingFace Hub client | ~1K | 90% |
| `hf_datasets_ex` | HuggingFace Datasets | ~2K | 85% |
| `tinkex` | Tinker platform client | ~8K | 95% |

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COOKBOOK LAYER                                   │
│    tinkex_cookbook    fireworks_cookbook    modal_cookbook              │
│    (adapter config only, inherits everything from crucible_kitchen)     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────────┐
│                      CRUCIBLE_KITCHEN                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Recipes   │  │  Workflows  │  │   Stages    │  │    Ports    │    │
│  │  (configs)  │  │  (control)  │  │  (actions)  │  │ (contracts) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────────┐
│                     FOUNDATION LAYER                                     │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │crucible_train│  │ hf_hub_ex  │  │hf_datasets_ex│  │crucible_     │   │
│  │  (renderers, │  │  (Hub API) │  │ (dataset    │  │model_registry│   │
│  │   datasets,  │  │            │  │  loading)   │  │              │   │
│  │   logging)   │  │            │  │             │  │              │   │
│  └──────────────┘  └────────────┘  └─────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKEND LAYER                                       │
│          tinkex (Tinker API)    |    Local Nx    |    Modal/Fireworks   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module-by-Module Mapping

### 3.1 Core Training Modules

#### `tinker_cookbook/supervised/train.py` (291 LOC)
| Component | Crucible Equivalent | Status | Notes |
|-----------|---------------------|--------|-------|
| `Config` class | `CrucibleKitchen.Recipes.SupervisedFinetuning` | EXISTS | Config schema |
| `main()` training loop | `CrucibleKitchen.Workflows.Supervised` | EXISTS | Workflow + Runner |
| `submit_batch()` | `CrucibleKitchen.Stages.ForwardBackward` | EXISTS | Stage implementation |
| `finish_batch()` | `CrucibleKitchen.Stages.LogStepMetrics` | EXISTS | |
| `run_evals()` | `CrucibleKitchen.Stages.Evaluate` | EXISTS | Needs enhancement |
| Resume logic | `CrucibleTrain.Checkpoint` | EXISTS | |
| Logging setup | `CrucibleTrain.Logging` | EXISTS | WandB, Neptune |

**Owner**: `crucible_kitchen` (workflow), `crucible_train` (implementation)
**tinkex_cookbook**: Recipe/config only

#### `tinker_cookbook/rl/train.py` (1118 LOC)
| Component | Crucible Equivalent | Status | Notes |
|-----------|---------------------|--------|-------|
| `Config` class | `CrucibleKitchen.Recipes.Reinforcement` | NEEDED | Recipe config |
| `do_sync_training()` | `CrucibleKitchen.Workflows.Reinforcement` | PLACEHOLDER | Need implementation |
| `do_async_training()` | `CrucibleKitchen.Workflows.Reinforcement` | PLACEHOLDER | Async variant |
| `train_step()` | `CrucibleTrain.RL.Train.train_step/4` | EXISTS | |
| `do_group_rollout()` | `CrucibleTrain.RL.Rollouts.do_group_rollout/2` | EXISTS | |
| PPO/IS loss | `CrucibleTrain.RL.Train` | EXISTS | |
| KL penalty | `CrucibleTrain.RL.Metrics` | EXISTS | |
| Advantage estimation | `CrucibleTrain.RL.DataProcessing` | EXISTS | |

**Owner**: `crucible_kitchen` (workflow), `crucible_train` (RL module)
**tinkex_cookbook**: Recipe/config only

#### `tinker_cookbook/preference/train_dpo.py` (396 LOC)
| Component | Crucible Equivalent | Status | Notes |
|-----------|---------------------|--------|-------|
| `Config` class | `CrucibleKitchen.Recipes.Preference` | NEEDED | |
| DPO loss computation | `CrucibleTrain.Preference.TrainDPO` | EXISTS | |
| Reference model handling | `CrucibleTrain.Preference.TrainDPO` | EXISTS | |
| `do_update()` | `CrucibleKitchen.Workflows.Preference` | PLACEHOLDER | |

**Owner**: `crucible_kitchen` (workflow), `crucible_train` (preference module)

#### `tinker_cookbook/distillation/train_on_policy.py` (462 LOC)
| Component | Crucible Equivalent | Status | Notes |
|-----------|---------------------|--------|-------|
| Multi-teacher support | `CrucibleTrain.Distillation.TrainOnPolicy` | EXISTS | |
| KL penalty with teacher | `CrucibleTrain.Distillation.TrainOnPolicy` | EXISTS | |
| Composite datasets | `CrucibleTrain.Distillation.Datasets` | EXISTS | |

**Owner**: `crucible_kitchen` (workflow), `crucible_train` (distillation module)

### 3.2 Renderers

#### `tinker_cookbook/renderers.py` (730 LOC)
| Renderer | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `RoleColonRenderer` | `CrucibleTrain.Renderers.RoleColon` | EXISTS |
| `Llama3Renderer` | `CrucibleTrain.Renderers.Llama3` | EXISTS |
| `Qwen3Renderer` | `CrucibleTrain.Renderers.Qwen3` | EXISTS |
| `DeepSeekV3Renderer` | `CrucibleTrain.Renderers.DeepseekV3` | EXISTS |
| `GptOssRenderer` | `CrucibleTrain.Renderers.GptOss` | EXISTS |
| `KimiK2Renderer` | `CrucibleTrain.Renderers.KimiK2` | EXISTS |
| `TrainOnWhat` enum | `CrucibleTrain.Renderers.TrainOnWhat` | EXISTS |
| `build_supervised_example()` | `CrucibleTrain.Renderers.Renderer.build_supervised_example/3` | EXISTS |

**Owner**: `crucible_train` (Renderers module)
**100% coverage** - all renderers ported

### 3.3 RL Components

#### `tinker_cookbook/rl/problem_env.py` (103 LOC)
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `ProblemEnv` | `CrucibleTrain.RL.ProblemEnv` | EXISTS |
| `ProblemGroupBuilder` | `CrucibleTrain.RL.EnvGroupBuilder` | EXISTS |

#### `tinker_cookbook/rl/types.py`
| Type | Crucible Equivalent | Status |
|------|---------------------|--------|
| `Env` behaviour | `CrucibleTrain.RL.Env` | EXISTS |
| `EnvGroupBuilder` | `CrucibleTrain.RL.EnvGroupBuilder` | EXISTS |
| `Trajectory` | `CrucibleTrain.RL.Types.Trajectory` | EXISTS |
| `TrajectoryGroup` | `CrucibleTrain.RL.Types.TrajectoryGroup` | EXISTS |

#### `tinker_cookbook/rl/rollouts.py`
| Function | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `do_group_rollout()` | `CrucibleTrain.RL.Rollouts.do_group_rollout/2` | EXISTS |

#### `tinker_cookbook/rl/data_processing.py`
| Function | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `compute_advantages()` | `CrucibleTrain.RL.DataProcessing.compute_advantages/1` | EXISTS |
| `assemble_training_data()` | `CrucibleTrain.RL.DataProcessing.assemble_training_data/2` | EXISTS |
| `remove_constant_reward_groups()` | `CrucibleTrain.RL.DataProcessing.remove_constant_reward_groups/1` | EXISTS |

#### `tinker_cookbook/rl/metrics.py`
| Function | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `compute_kl_sample_train()` | `CrucibleTrain.RL.Metrics` | EXISTS |
| `compute_post_kl()` | `CrucibleTrain.RL.Metrics` | EXISTS |
| `incorporate_kl_penalty()` | `CrucibleTrain.RL.Metrics` | EXISTS |

### 3.4 Completers

#### `tinker_cookbook/completers.py` (114 LOC)
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `TokenCompleter` behaviour | `CrucibleTrain.Completers.TokenCompleter` | EXISTS |
| `MessageCompleter` behaviour | `CrucibleTrain.Completers.MessageCompleter` | EXISTS |
| `TinkerTokenCompleter` | `crucible_kitchen` adapter | NEEDED in crucible_kitchen |
| `TinkerMessageCompleter` | `crucible_kitchen` adapter | NEEDED in crucible_kitchen |

**Owner**: `crucible_train` (behaviours), `crucible_kitchen` (Tinkex adapters)

### 3.5 Evaluation

#### `tinker_cookbook/eval/evaluators.py`
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `Evaluator` base | `CrucibleTrain.Eval.Evaluator` | EXISTS |
| `TrainingClientEvaluator` | `CrucibleTrain.Eval.Evaluator` | EXISTS |
| `SamplingClientEvaluator` | `CrucibleTrain.Eval.Evaluator` | EXISTS |
| `EvaluatorBuilder` | `CrucibleTrain.Eval.Evaluators` | EXISTS |

#### `tinker_cookbook/supervised/nll_evaluator.py`
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `NLLEvaluator` | `CrucibleTrain.Supervised.NLLEvaluator` | EXISTS |

#### `tinker_cookbook/eval/scorers/`
| Scorer | Crucible Equivalent | Status |
|--------|---------------------|--------|
| `exact_match` | `CrucibleTrain.Eval.Scorers.ExactMatch` | EXISTS |
| `contains` | `CrucibleTrain.Eval.Scorers.Contains` | EXISTS |
| `semantic_similarity` | `CrucibleTrain.Eval.Scorers.SemanticSimilarity` | EXISTS |

### 3.6 Utilities

#### `tinker_cookbook/utils/lr_scheduling.py`
| Function | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `compute_schedule_lr_multiplier()` | `CrucibleTrain.Utils.LRScheduling` | EXISTS |
| Linear schedule | `CrucibleKitchen.LRSchedule.linear/3` | EXISTS |
| Cosine schedule | `CrucibleKitchen.LRSchedule.cosine/3` | EXISTS |
| Warmup | `CrucibleKitchen.LRSchedule.with_warmup/3` | EXISTS |

#### `tinker_cookbook/utils/logtree.py`
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `LogTree` | `CrucibleTrain.Utils.Logtree` | EXISTS |
| HTML export | `CrucibleTrain.Utils.LogtreeFormatters` | EXISTS |

#### `tinker_cookbook/utils/trace.py`
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `@scope` decorator | `CrucibleTrain.Utils.Trace` | EXISTS |
| Trace export | `CrucibleTrain.Utils.Trace` | EXISTS |

#### `tinker_cookbook/utils/ml_log.py`
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| WandB logger | `CrucibleTrain.Logging.WandbLogger` | EXISTS |
| Neptune logger | `CrucibleTrain.Logging.NeptuneLogger` | EXISTS |
| JSON logger | `CrucibleTrain.Logging.JsonLogger` | EXISTS |
| Multiplexed logging | `CrucibleTrain.Logging.MultiplexLogger` | EXISTS |

#### `tinker_cookbook/checkpoint_utils.py`
| Function | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `save_checkpoint()` | `CrucibleTrain.Checkpoint.save/3` | EXISTS |
| `get_last_checkpoint()` | `CrucibleTrain.Checkpoint.latest/1` | EXISTS |
| Resume state | `CrucibleTrain.Checkpoint` | EXISTS |

### 3.7 Data/Types

#### `tinker_cookbook/supervised/data.py` & `types.py`
| Type | Crucible Equivalent | Status |
|------|---------------------|--------|
| `SupervisedDataset` | `CrucibleTrain.Supervised.Dataset` | EXISTS |
| `SupervisedDatasetBuilder` | `CrucibleKitchen.DatasetBuilder` | EXISTS |
| `ChatDatasetBuilder` | `CrucibleKitchen.DatasetBuilder` | EXISTS |

#### `tinker_cookbook/preference/types.py` & `dpo_datasets.py`
| Type | Crucible Equivalent | Status |
|------|---------------------|--------|
| `PreferenceDataset` | `CrucibleTrain.Preference.DPODatasets` | EXISTS |
| `LabeledComparison` | `CrucibleTrain.Preference.Types` | EXISTS |

#### Core types from tinker SDK
| Type | Crucible Equivalent | Status |
|------|---------------------|--------|
| `Datum` | `CrucibleTrain.Types.Datum` | EXISTS |
| `ModelInput` | `CrucibleTrain.Types.ModelInput` | EXISTS |
| `TensorData` | `CrucibleTrain.Types.TensorData` | EXISTS |
| `TokensWithLogprobs` | `CrucibleTrain.Types.TokensWithLogprobs` | EXISTS |

### 3.8 Model Info & Hyperparams

#### `tinker_cookbook/model_info.py`
| Function | Crucible Equivalent | Status |
|----------|---------------------|--------|
| `get_model_info()` | `CrucibleTrain.ModelInfo.get/1` | EXISTS |
| Renderer selection | `CrucibleTrain.Renderers.Registry.get_renderer/1` | EXISTS |

#### `tinker_cookbook/hyperparam_utils.py`
| Component | Crucible Equivalent | Status |
|-----------|---------------------|--------|
| `chz.chz` configs | `CrucibleKitchen.Hyperparam` | EXISTS |

### 3.9 Recipes (Example Scripts)

#### `tinker_cookbook/recipes/` directory
| Recipe | Crucible Equivalent | Status |
|--------|---------------------|--------|
| `sl_basic.py` | `CrucibleKitchen.Recipes.SupervisedFinetuning` | EXISTS |
| `sl_loop.py` | Example in docs | N/A |
| `rl_basic.py` | `CrucibleKitchen.Recipes.Reinforcement` | NEEDED |
| `rl_loop.py` | Example in docs | N/A |
| `preference/dpo/train.py` | `CrucibleKitchen.Recipes.Preference` | NEEDED |
| `distillation/on_policy_distillation.py` | `CrucibleKitchen.Recipes.Distillation` | NEEDED |
| `math_rl/` | Example in tinkex_cookbook | NEEDED |
| `verifiers_rl/` | Example in tinkex_cookbook | NEEDED |
| `tool_use/search/` | Example in tinkex_cookbook | NEEDED |
| `multiplayer_rl/` | Example in tinkex_cookbook | NEEDED |

---

## 4. Dependency Diagram

### 4.1 tinkex_cookbook Dependency Flow

```
                        tinkex_cookbook
                              |
                              | (provides adapters)
                              v
                      ┌───────────────────┐
                      │  CrucibleKitchen  │
                      │                   │
                      │  - Workflows      │
                      │  - Stages         │
                      │  - Ports          │
                      │  - Recipes        │
                      └─────────┬─────────┘
                                |
        ┌───────────┬───────────┼───────────┬───────────┐
        v           v           v           v           v
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │ tinkex │  │crucible│  │hf_hub_ex│ │hf_data-│  │crucible│
   │        │  │_train  │  │        │  │sets_ex │  │_model_ │
   │        │  │        │  │        │  │        │  │registry│
   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
```

### 4.2 What Goes Through crucible_kitchen vs Direct Calls

| Operation | Route | Reason |
|-----------|-------|--------|
| Training orchestration | crucible_kitchen | Workflow control |
| forward/backward calls | crucible_kitchen -> tinkex | Via TrainingClient port |
| Dataset loading | crucible_kitchen -> hf_datasets_ex | Via DatasetStore port |
| Checkpointing | crucible_kitchen -> tinkex | Via BlobStore port |
| Tokenization | adapter-specific | TrainingClient helper or tiktoken_ex |
| Model registry | Direct tinkex call | Not orchestrated |
| Raw API access | Direct tinkex call | Debugging/advanced |

### 4.3 Adapter Mapping

```elixir
# CrucibleKitchen adapters (referenced by tinkex_cookbook recipes)
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []},
  blob_store: {CrucibleKitchen.Adapters.Noop.BlobStore, []},
  hub_client: {CrucibleKitchen.Adapters.HfHub.HubClient, []},
  metrics_store: {CrucibleTelemetry.Adapters.JSONLMetrics, path: "/tmp/metrics.jsonl"},
  completer: {CrucibleKitchen.Adapters.Tinkex.Completer, []}
}
```

---

## 5. Gap Analysis

### 5.1 Fully Covered (No Work Needed)

- All renderers (Llama3, Qwen3, DeepSeekV3, GPT-OSS, RoleColon, KimiK2)
- Supervised training infrastructure
- DPO/preference training core
- RL core (rollouts, advantages, metrics)
- Distillation core
- Logging (WandB, Neptune, JSON)
- LR scheduling
- Checkpoint management
- Type definitions (Datum, ModelInput, TensorData)
- Evaluators and scorers

### 5.2 Needs Enhancement in crucible_kitchen

| Gap | Location | Effort |
|-----|----------|--------|
| RL workflow placeholder | `CrucibleKitchen.Workflows.Reinforcement` | M |
| Preference workflow placeholder | `CrucibleKitchen.Workflows.Preference` | M |
| Distillation workflow placeholder | `CrucibleKitchen.Workflows.Distillation` | M |
| Async training support | `CrucibleKitchen.Workflow.Runner` | L |
| Stream minibatch config | `CrucibleKitchen.Workflows.Reinforcement` | S |

### 5.3 Needs Implementation in crucible_kitchen

| Component | Purpose | Effort |
|-----------|---------|--------|
| `CrucibleKitchen.Adapters.Tinkex.TrainingClient` | Bridge tinkex -> CrucibleTrain.Ports.TrainingClient | S |
| `CrucibleKitchen.Adapters.HfDatasets.DatasetStore` | Bridge hf_datasets_ex -> CrucibleTrain.Ports.DatasetStore | S |
| `CrucibleKitchen.Adapters.Noop.BlobStore` | Bridge blob ops -> CrucibleTrain.Ports.BlobStore | S |
| `CrucibleKitchen.Adapters.Tinkex.Completer` | Bridge tinkex SamplingClient -> CrucibleKitchen.Ports.Completer | S |
| Tokenizer helpers (optional) | Adapter-specific tokenizer access | S |
| Example recipes | Math RL, tool use, multiplayer | M |

---

## 6. New Projects Needed

### 6.1 No New Projects Required

The existing ecosystem covers all functionality. The only "new" code is:

1. **crucible_kitchen adapters** (~200 LOC) - Wrap external SDKs
2. **crucible_kitchen workflow implementations** (~500 LOC) - Within existing project

### 6.2 Recommended NOT to Create

| Idea | Reason to Avoid |
|------|-----------------|
| `crucible_renderers` | Already in `crucible_train.Renderers` |
| `crucible_completers` | Already in `crucible_train.Completers` |
| `crucible_rl` | Already in `crucible_train.RL` |
| `crucible_eval` | Already in `crucible_train.Eval` |

---

## 7. Interface Design

### 7.1 tinkex_cookbook Public API

```elixir
# Entry point - that's it!
TinkexCookbook.run(:supervised, %{
  model: "meta-llama/Llama-3.1-8B-Instruct",
  dataset: "HuggingFaceH4/no_robots",
  epochs: 3,
  learning_rate: 2.0e-4,
  lora_rank: 32
})

TinkexCookbook.run(:reinforcement, %{
  model: "meta-llama/Llama-3.1-8B-Instruct",
  env: MyEnv,
  max_tokens: 1024,
  batches: 100
})

TinkexCookbook.run(:preference, %{
  model: "meta-llama/Llama-3.1-8B-Instruct",
  dataset: "argilla/ultrafeedback-binarized-preferences",
  dpo_beta: 0.1
})

TinkexCookbook.run(:distillation, %{
  student_model: "meta-llama/Llama-3.1-8B-Instruct",
  teacher_model: "meta-llama/Llama-3.1-70B-Instruct",
  dataset: "gsm8k"
})
```

### 7.2 Adapter Implementation Example

```elixir
defmodule CrucibleKitchen.Adapters.Tinkex.TrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient

  @impl true
  def start_session(opts, config) do
    # Use tinkex to create training client
    with {:ok, service} <- Tinkex.ServiceClient.start_link(opts),
         {:ok, client} <- Tinkex.ServiceClient.create_lora_training_client(
           service,
           config.model,
           rank: config.lora_rank
         ) do
      {:ok, %{service: service, client: client}}
    end
  end

  @impl true
  def forward_backward(opts, session, datums) do
    Tinkex.TrainingClient.forward_backward(
      session.client,
      datums,
      :cross_entropy,
      opts
    )
  end

  @impl true
  def optim_step(opts, session, learning_rate) do
    adam_params = %{learning_rate: learning_rate, beta1: 0.9, beta2: 0.95, eps: 1.0e-8}
    Tinkex.TrainingClient.optim_step(session.client, adam_params, opts)
  end

  @impl true
  def await(opts, future) do
    Task.await(future, Keyword.get(opts, :timeout, :infinity))
  end

  @impl true
  def save_checkpoint(opts, session, path) do
    {:ok, task} = Tinkex.TrainingClient.save_state(session.client, path, opts)
    Task.await(task)
  end

  @impl true
  def close_session(_opts, session) do
    Tinkex.TrainingClient.unload_model(session.client)
  end
end
```

### 7.3 Completer Adapter

```elixir
defmodule CrucibleKitchen.Adapters.Tinkex.Completer do
  @behaviour CrucibleKitchen.Ports.Completer

  @impl true
  def complete(opts, messages, completion_opts) do
    sampling_client = Keyword.fetch!(opts, :sampling_client)
    renderer = Keyword.fetch!(opts, :renderer)
    model_input = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()

    sampling_params = %{
      max_tokens: Keyword.get(completion_opts, :max_tokens, 1024),
      temperature: Keyword.get(completion_opts, :temperature, 1.0),
      stop: stop
    }

    with {:ok, result} <- Tinkex.SamplingClient.sample(
           sampling_client,
           model_input,
           1,
           sampling_params
         ) do
      sequence = hd(result.sequences)
      {text, _parsed?} = renderer.parse_response(sequence.tokens)

      {:ok,
       %{
         text: text,
         finish_reason: sequence.finish_reason,
         usage: result.usage,
         logprobs: sequence.logprobs,
         tool_calls: nil
       }}
    end
  end

  @impl true
  def complete_batch(opts, prompts, completion_opts) do
    prompts
    |> Enum.map(&complete(opts, &1, completion_opts))
    |> collect_results()
  end

  @impl true
  def supports_streaming?(_opts), do: false

  @impl true
  def supports_tools?(_opts), do: false

  @impl true
  def get_model(opts), do: Keyword.fetch!(opts, :model)

  defp collect_results(results) do
    case Enum.split_with(results, &match?({:ok, _}, &1)) do
      {oks, []} -> {:ok, Enum.map(oks, fn {:ok, result} -> result end)}
      {_oks, errors} -> {:error, errors}
    end
  end
end
```

---

## 8. Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Get supervised training working end-to-end

1. Implement `CrucibleKitchen.Adapters.Tinkex.TrainingClient`
2. Implement `CrucibleKitchen.Adapters.HfDatasets.DatasetStore`
3. Implement tokenizer helpers (optional)
4. Verify `CrucibleKitchen.Workflows.Supervised` works
5. Create integration test with Tinker backend

**Deliverable**: `TinkexCookbook.run(:supervised, config)` works

### Phase 2: RL & Preference (Week 2)

**Goal**: Get RL and DPO training working

1. Implement `CrucibleKitchen.Workflows.Reinforcement` (fill placeholder)
2. Implement `CrucibleKitchen.Workflows.Preference` (fill placeholder)
3. Implement `CrucibleKitchen.Adapters.Tinkex.Completer`
4. Implement `CrucibleKitchen.Adapters.Noop.BlobStore` or a real blob adapter
5. Port math_rl example

**Deliverable**: `TinkexCookbook.run(:reinforcement, config)` and `TinkexCookbook.run(:preference, config)` work

### Phase 3: Distillation & Advanced (Week 3)

**Goal**: Complete feature parity

1. Implement `CrucibleKitchen.Workflows.Distillation` (fill placeholder)
2. Add async training support to Runner
3. Add stream minibatch support
4. Port tool_use/search example
5. Port multiplayer_rl example

**Deliverable**: Full feature parity with tinker-cookbook

### Phase 4: Polish & Documentation (Week 4)

**Goal**: Production-ready

1. Performance optimization
2. Error handling improvements
3. Telemetry enhancements
4. Documentation (getting started, examples)
5. Property-based tests

**Deliverable**: Production-ready tinkex_cookbook

---

## Appendix A: File Counts

### tinker-cookbook (Python)
```
tinker_cookbook/
├── supervised/     ~400 LOC (train, data, types, nll_evaluator)
├── rl/            ~800 LOC (train, rollouts, types, metrics, data_processing)
├── preference/    ~500 LOC (train_dpo, dpo_datasets, types)
├── distillation/  ~300 LOC (train_on_policy, datasets)
├── eval/          ~400 LOC (evaluators, scorers)
├── utils/         ~600 LOC (ml_log, logtree, trace, lr_scheduling)
├── recipes/       ~2000 LOC (examples, domain-specific)
├── renderers.py   ~730 LOC
├── completers.py  ~114 LOC
├── other          ~500 LOC (model_info, checkpoint_utils, display, etc.)
Total: ~6,344 LOC (core) + ~2,000 LOC (recipes)
```

### crucible_ ecosystem (Elixir)
```
crucible_train/    ~5,000 LOC (80% of tinker-cookbook core)
crucible_kitchen/  ~2,000 LOC (orchestration)
tinkex/            ~8,000 LOC (Tinker API client)
hf_datasets_ex/    ~2,000 LOC
hf_hub_ex/         ~1,000 LOC
crucible_model_registry/ ~1,500 LOC
```

### tinkex_cookbook (Target)
```
tinkex_cookbook/
├── adapters/      ~200 LOC (5 adapters, ~40 LOC each)
├── recipes/       ~200 LOC (4 recipe configs)
└── examples/      ~300 LOC (domain examples)
Total: ~700 LOC
```

---

## Appendix B: Complete tinker-cookbook Module Inventory

| Module | LOC | Crucible Owner | Status |
|--------|-----|----------------|--------|
| `supervised/train.py` | 291 | crucible_kitchen + crucible_train | EXISTS |
| `supervised/data.py` | 150 | crucible_train | EXISTS |
| `supervised/types.py` | 50 | crucible_train | EXISTS |
| `supervised/common.py` | 30 | crucible_train | EXISTS |
| `supervised/nll_evaluator.py` | 80 | crucible_train | EXISTS |
| `rl/train.py` | 1118 | crucible_kitchen + crucible_train | PARTIAL |
| `rl/types.py` | 120 | crucible_train | EXISTS |
| `rl/rollouts.py` | 200 | crucible_train | EXISTS |
| `rl/data_processing.py` | 150 | crucible_train | EXISTS |
| `rl/metrics.py` | 200 | crucible_train | EXISTS |
| `rl/problem_env.py` | 103 | crucible_train | EXISTS |
| `rl/metric_util.py` | 100 | crucible_train | EXISTS |
| `rl/preference_envs.py` | 100 | crucible_train | EXISTS |
| `preference/train_dpo.py` | 396 | crucible_kitchen + crucible_train | EXISTS |
| `preference/dpo_datasets.py` | 150 | crucible_train | EXISTS |
| `preference/types.py` | 50 | crucible_train | EXISTS |
| `preference/preference_datasets.py` | 100 | crucible_train | EXISTS |
| `preference/comparison_policy_evaluator.py` | 100 | crucible_train | EXISTS |
| `distillation/train_on_policy.py` | 462 | crucible_kitchen + crucible_train | EXISTS |
| `distillation/datasets.py` | 100 | crucible_train | EXISTS |
| `renderers.py` | 730 | crucible_train | EXISTS |
| `completers.py` | 114 | crucible_train + crucible_kitchen | PARTIAL |
| `eval/evaluators.py` | 150 | crucible_train | EXISTS |
| `eval/custom_evaluators.py` | 100 | crucible_train | EXISTS |
| `eval/scorers/*` | 150 | crucible_train | EXISTS |
| `utils/ml_log.py` | 200 | crucible_train | EXISTS |
| `utils/logtree.py` | 150 | crucible_train | EXISTS |
| `utils/trace.py` | 100 | crucible_train | EXISTS |
| `utils/lr_scheduling.py` | 50 | crucible_kitchen + crucible_train | EXISTS |
| `utils/misc_utils.py` | 100 | crucible_train | EXISTS |
| `checkpoint_utils.py` | 100 | crucible_train | EXISTS |
| `model_info.py` | 80 | crucible_train | EXISTS |
| `hyperparam_utils.py` | 50 | crucible_kitchen | EXISTS |
| `tokenizer_utils.py` | 80 | tinkex | EXISTS |
| `display.py` | 50 | crucible_train | EXISTS |
| `cli_utils.py` | 100 | N/A (CLI) | NOT NEEDED |
| `recipes/*` | 2000 | tinkex_cookbook (examples) | PARTIAL |
| `tests/*` | 500 | Tests | SEPARATE |
| `chat_app/*` | 150 | N/A (demo) | NOT NEEDED |

---

## Appendix C: Port Behaviour Definitions

### CrucibleTrain.Ports.TrainingClient

```elixir
@callback start_session(opts :: keyword(), config :: map()) ::
  {:ok, session()} | {:error, term()}

@callback forward_backward(opts :: keyword(), session(), [Datum.t()]) :: future()

@callback optim_step(opts :: keyword(), session(), learning_rate :: float()) :: future()

@callback await(opts :: keyword(), future()) :: {:ok, map()} | {:error, term()}

@callback save_checkpoint(opts :: keyword(), session(), path :: String.t()) :: :ok | {:error, term()}

@callback load_checkpoint(opts :: keyword(), session(), path :: String.t()) :: :ok | {:error, term()}

@callback close_session(opts :: keyword(), session()) :: :ok
```

### CrucibleKitchen.Ports.Completer

```elixir
@callback complete(opts :: keyword(), [Message.t() | map()], completion_opts()) ::
  {:ok, completion_result()} | {:error, term()}

@callback complete_batch(opts :: keyword(), [[Message.t() | map()]], completion_opts()) ::
  {:ok, [completion_result()]} | {:error, term()}

@callback stream_complete(opts :: keyword(), [Message.t() | map()], completion_opts()) ::
  {:ok, Enumerable.t()} | {:error, term()}

@callback supports_streaming?(opts :: keyword()) :: boolean()

@callback supports_tools?(opts :: keyword()) :: boolean()

@callback get_model(opts :: keyword()) :: String.t()
```

### CrucibleTrain.Ports.DatasetStore

```elixir
@callback load_dataset(opts :: keyword(), repo_id :: String.t(), load_opts :: keyword()) ::
  {:ok, dataset()} | {:error, term()}

@callback get_split(opts :: keyword(), dataset(), split :: String.t() | atom()) ::
  {:ok, dataset()} | {:error, term()}

@callback shuffle(opts :: keyword(), dataset(), shuffle_opts :: keyword()) ::
  {:ok, dataset()} | {:error, term()}

@callback take(opts :: keyword(), dataset(), count :: non_neg_integer()) ::
  {:ok, dataset()} | {:error, term()}

@callback skip(opts :: keyword(), dataset(), count :: non_neg_integer()) ::
  {:ok, dataset()} | {:error, term()}

@callback select(opts :: keyword(), dataset(), selection :: Range.t() | [non_neg_integer()]) ::
  {:ok, dataset()} | {:error, term()}

@callback to_list(opts :: keyword(), dataset()) ::
  {:ok, [map()]} | {:error, term()}
```

---

## Summary

**tinkex_cookbook is NOT a port of tinker-cookbook.** It is a thin (~700 LOC) configuration layer that:

1. Defines recipe configurations for common workflows
2. References adapters implemented in `crucible_kitchen`
3. Contains example domain-specific recipes (math RL, tool use, etc.)

All training logic, renderers, evaluators, utilities, and orchestration already exist in:
- `crucible_kitchen` - Workflow orchestration
- `crucible_train` - Training infrastructure
- `tinkex` - Tinker API client
- `hf_datasets_ex` - Dataset loading
- `hf_hub_ex` - Hub integration
- `crucible_model_registry` - Model versioning

This requires adapters in `crucible_kitchen` and workflow implementations in `crucible_kitchen` (filling existing placeholders).
