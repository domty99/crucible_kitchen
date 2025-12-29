# Training Abstractions in tinker-cookbook

Comprehensive analysis of training-related code in the tinker-cookbook Python library for mapping to the Elixir crucible ecosystem (crucible_kitchen, crucible_train, tinkex).

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Training Loop Abstractions](#training-loop-abstractions)
4. [Gradient Accumulation and Substeps](#gradient-accumulation-and-substeps)
5. [Distributed/Async Training](#distributedasync-training)
6. [Checkpointing and Resumption](#checkpointing-and-resumption)
7. [Mixed Precision Training](#mixed-precision-training)
8. [LoRA/Adapter Training](#loraadapter-training)
9. [Optimizer Configurations](#optimizer-configurations)
10. [Learning Rate Scheduling](#learning-rate-scheduling)
11. [Loss Functions](#loss-functions)
12. [Data Processing Abstractions](#data-processing-abstractions)
13. [Mapping to Elixir Ecosystem](#mapping-to-elixir-ecosystem)
14. [Implementation Status](#implementation-status)
15. [Recommendations](#recommendations)

---

## Executive Summary

The tinker-cookbook library provides a comprehensive training framework built on the Tinker service (hosted by Thinking Machines Lab). Key architectural insight: **training loops run on CPU clients while GPU-heavy work (forward/backward, sampling) is offloaded to the Tinker service**.

### Key Training Paradigms Supported:
- **Supervised Learning (SL)** - Cross-entropy loss on labeled data
- **Reinforcement Learning (RL)** - GRPO-style policy gradient with importance sampling
- **Direct Preference Optimization (DPO)** - Preference learning from comparisons
- **On-Policy Distillation** - KL-based knowledge transfer from teachers
- **Multi-task Training** - Composite datasets mixing environments

### Core Abstractions:
| Python Abstraction | Purpose | Elixir Target |
|-------------------|---------|---------------|
| `TrainingClient` | Forward/backward, optimizer steps | `Tinkex.TrainingClient` |
| `SamplingClient` | Model inference during RL | `Tinkex.SamplingClient` |
| `SupervisedDataset` | Batched training data | `CrucibleKitchen.Dataset` |
| `RLDataset` / `EnvGroupBuilder` | RL environments | `CrucibleKitchen.RLEnv` |
| `checkpoint_utils` | Save/resume training | `CrucibleKitchen.Checkpoint` |
| `lr_scheduling` | LR decay strategies | `CrucibleKitchen.Scheduler` |

---

## Architecture Overview

### Client-Server Split

```
+------------------+     HTTP/REST      +-------------------+
|  Client (CPU)    | <--------------->  |  Tinker Service   |
|                  |                    |  (GPU Cluster)    |
+------------------+                    +-------------------+
|                  |                    |                   |
| - Data loading   |  forward_backward  | - Forward pass    |
| - Tokenization   | <--------------->  | - Backward pass   |
| - Reward compute |                    | - Gradient accum  |
| - Advantage calc |  optim_step        | - Optimizer step  |
| - Checkpointing  | <--------------->  | - Weight storage  |
| - Logging        |                    |                   |
+------------------+                    +-------------------+
```

### Key Files Structure

```
tinker_cookbook/
├── supervised/
│   ├── train.py           # Main SL training loop
│   ├── common.py          # Loss computation, datum helpers
│   ├── types.py           # SupervisedDataset, ChatDatasetBuilder
│   └── data.py            # Data conversion utilities
├── rl/
│   ├── train.py           # Main RL training loop
│   ├── types.py           # Env, EnvGroupBuilder, RLDataset
│   ├── data_processing.py # Trajectory -> Datum conversion
│   ├── rollouts.py        # Rollout collection
│   └── metrics.py         # KL computation, reward metrics
├── preference/
│   ├── train_dpo.py       # DPO training
│   ├── types.py           # Comparison, PreferenceModel
│   └── dpo_datasets.py    # DPO dataset builders
├── distillation/
│   ├── train_on_policy.py # On-policy distillation
│   └── datasets.py        # CompositeDataset, TeacherConfig
├── checkpoint_utils.py    # Checkpointing
├── hyperparam_utils.py    # LR scaling, param counting
└── utils/
    ├── lr_scheduling.py   # LR schedules
    └── ml_log.py          # Logging infrastructure
```

---

## Training Loop Abstractions

### 1. Supervised Learning Loop

**File:** `tinker_cookbook/supervised/train.py`

The SL loop implements **pipelined training** where the next batch is submitted before the previous one completes.

```python
# Core training loop structure
async def main(config: Config):
    # 1. Resume handling
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)

    # 2. Client initialization
    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_info["state_path"]
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank
        )

    # 3. Pipelined batch processing
    pending_batch = None
    for epoch_idx in range(start_epoch, config.num_epochs):
        dataset.set_epoch(seed=epoch_idx)  # Shuffle per epoch

        for batch_idx in range(start_batch_idx, n_batches):
            submitted_batch = await submit_batch(epoch_idx, batch_idx)
            if pending_batch is not None:
                await finish_batch(pending_batch)  # Process previous
            pending_batch = submitted_batch
```

**Key Pattern:** Double-buffering where `submit_batch` sends requests and `finish_batch` awaits results.

```python
async def submit_batch(epoch_idx, batch_idx):
    # Calculate LR with schedule
    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(...)

    # Run evals BEFORE training step (captures pre-step weights)
    if evaluators and step % config.eval_every == 0:
        eval_metrics = await run_evals(evaluators, training_client, step)

    # Submit forward-backward and optimizer step (pipelined)
    fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
    optim_step_future = await training_client.optim_step_async(adam_params)

    return SubmittedBatch(fwd_bwd_future, optim_step_future, ...)
```

### 2. RL Training Loop

**File:** `tinker_cookbook/rl/train.py`

RL training supports three modes:
1. **Synchronous on-policy** (`do_sync_training`)
2. **Synchronous with minibatch streaming** (`do_sync_training_with_stream_minibatch`)
3. **Asynchronous off-policy** (`do_async_training`)

```python
async def main(cfg: Config):
    # Mode selection
    if cfg.async_config is not None:
        training_func = do_async_training
    elif cfg.stream_minibatch_config is not None:
        training_func = do_sync_training_with_stream_minibatch
    else:
        training_func = do_sync_training
```

**Synchronous RL Pattern:**

```python
async def do_sync_training(...):
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(...)

    for i_batch in range(start_batch, end_batch):
        # 1. Get environments for this batch
        env_group_builders_P = dataset.get_batch(i_batch)

        # 2. Parallel rollout collection
        trajectory_groups_P = await asyncio.gather(*[
            do_group_rollout_and_filter_constant_reward(
                sampling_client, builder, max_tokens, temperature, ...
            )
            for builder in env_group_builders_P
        ])

        # 3. Training step
        sampling_client, metrics = await do_train_step_and_get_sampling_client(
            cfg, i_batch, training_client, ..., trajectory_groups_P
        )
```

**RL Train Step:**

```python
async def train_step(data_D, training_client, learning_rate, num_substeps, loss_fn):
    """Pipelines forward_backward and optim_step to same clock cycle."""
    batches = split_list(data_D, min(num_substeps, len(data_D)))

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(batches[0], loss_fn=loss_fn)
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue NEXT batch before consuming current (same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(batches[i + 1], ...)
            next_optim_future = await training_client.optim_step_async(adam_params)

        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
```

### 3. DPO Training Loop

**File:** `tinker_cookbook/preference/train_dpo.py`

DPO uses `forward_backward_custom` for flexible loss computation:

```python
def do_update(...):
    # Split data into chosen/rejected pairs
    chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
    rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

    # Get reference logprobs from frozen reference model
    all_ref_logprobs = asyncio.run(compute_all_ref_logprobs())

    # Custom DPO loss function
    def dpo_loss_fn(data, logprobs_list):
        # Compute log ratios: log(p_theta/p_ref) for chosen and rejected
        chosen_log_ratio = torch.stack([lp - rlp for lp, rlp in zip(...)])
        rejected_log_ratio = torch.stack([lp - rlp for lp, rlp in zip(...)])

        # DPO loss: -log sigmoid(beta * (chosen_log_ratio - rejected_log_ratio))
        losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
        return losses.mean(), {"dpo_loss": ..., "accuracy": ..., "margin": ...}

    # Execute with custom loss
    backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
    training_client.optim_step(adam_params).result()
```

---

## Gradient Accumulation and Substeps

**Substep Mechanism:** Large batches are split into multiple forward_backward calls, each followed by optim_step.

```python
@chz.chz
class Config:
    num_substeps: int = 1  # Number of optimizer steps per training iteration
```

**Implementation in RL:**

```python
async def train_step(data_D, training_client, learning_rate, num_substeps, loss_fn):
    batches = split_list(data_D, min(num_substeps, len(data_D)))

    for batch in batches:
        fwd_bwd_future = await training_client.forward_backward_async(batch, loss_fn=loss_fn)
        optim_future = await training_client.optim_step_async(adam_params)
        await fwd_bwd_future.result_async()
        await optim_future.result_async()
```

**Minibatch Streaming (RL):**

```python
@chz.chz
class StreamMinibatchConfig:
    groups_per_batch: int        # Total trajectory groups
    num_minibatches: int         # Minibatches per substep
```

This allows training on partial rollouts as they complete:

```python
async def do_train_step_streaming_and_get_sampling_client(...):
    for i_substep in range(cfg.num_substeps):
        forward_backward_futures = []

        while i_minibatch < cfg.stream_minibatch_config.num_minibatches:
            # Wait for enough trajectories
            wrapped_trajectory_group = await trajectory_groups_queue.get()

            if len(wrapped_trajectory_groups) >= groups_per_minibatch:
                # Enqueue forward-backward (don't await yet)
                forward_backward_futures.append(
                    await training_client.forward_backward_async(data_D, loss_fn=cfg.loss_fn)
                )

        # Enqueue optim_step before awaiting
        optim_future = await training_client.optim_step_async(adam_params)

        # Now consume all results
        for fwd_bwd_future in forward_backward_futures:
            await fwd_bwd_future.result_async()
        await optim_future.result_async()
```

---

## Distributed/Async Training

### Async Off-Policy Configuration

```python
@chz.chz
class AsyncConfig:
    max_steps_off_policy: int  # Skip stale samples
    groups_per_batch: int      # Minimum batch size to maintain
```

### Async Training Architecture

```python
async def do_async_training(...):
    # Queues for producer-consumer pattern
    env_group_builders_queue = asyncio.Queue(maxsize=cfg.async_config.groups_per_batch)
    trajectory_groups_queue = asyncio.Queue()

    # Four concurrent loops
    await asyncio.gather(
        asyncio.create_task(dataloader_loop()),           # Produces env builders
        *[asyncio.create_task(trajectory_group_worker_loop())  # Workers sample
          for _ in range(cfg.async_config.groups_per_batch)],
        asyncio.create_task(training_loop()),             # Consumes trajectories
        asyncio.create_task(evaluation_loop()),           # Periodic evals
    )
```

**Staleness Filtering:**

```python
def filter_stale_trajectory_group(wrapped_trajectory_group):
    if i_batch - wrapped_trajectory_group.sampling_client_step > cfg.async_config.max_steps_off_policy:
        # Requeue the environment for resampling
        asyncio.create_task(env_group_builders_queue.put(wrapped_trajectory_group.env_group_builder))
        return False  # Skip this sample
    return True
```

---

## Checkpointing and Resumption

**File:** `tinker_cookbook/checkpoint_utils.py`

### Checkpoint Types

```python
kind: Literal["state", "sampler", "both"]
# - "state": Full optimizer state (for resumption)
# - "sampler": Just weights (for inference)
# - "both": Both types
```

### Save Checkpoint

```python
async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],  # {"epoch": 2, "batch": 150}
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}

    # Append to checkpoints.jsonl
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths
```

### Resume Training

```python
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict | None:
    """Get last checkpoint with full resumable state."""
    checkpoints = read_jsonl(os.path.join(log_dir, "checkpoints.jsonl"))
    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    return checkpoints_with_key[-1] if checkpoints_with_key else None
```

### Usage in Training Loops

```python
# Check for resume
resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
if resume_info:
    # Resuming: load optimizer state
    training_client = await service_client.create_training_client_from_state_with_optimizer_async(
        resume_info["state_path"]
    )
    start_epoch = resume_info["epoch"]
    start_batch = resume_info["batch"]
elif config.load_checkpoint_path:
    # Starting fresh from checkpoint: weights only (fresh optimizer)
    training_client = await service_client.create_training_client_from_state_async(
        config.load_checkpoint_path
    )
else:
    # Fresh start
    training_client = await service_client.create_lora_training_client_async(...)
```

---

## Mixed Precision Training

**Note:** Mixed precision is handled server-side by the Tinker service. The client library does not expose explicit mixed precision controls.

The service automatically uses appropriate precision for:
- Forward pass
- Backward pass
- Optimizer state
- Weight storage

---

## LoRA/Adapter Training

### LoRA Configuration

```python
training_client = await service_client.create_lora_training_client_async(
    base_model=config.model_name,  # "meta-llama/Llama-3.1-8B"
    rank=config.lora_rank,         # Default: 32
    user_metadata={"wandb_link": ...}
)
```

### LoRA LR Scaling

**File:** `tinker_cookbook/hyperparam_utils.py`

```python
def get_lora_lr_over_full_finetune_lr(model_name: str, lora_alpha: int = 32) -> float:
    """LoRA needs ~10x higher LR than full fine-tuning."""
    return 10.0

def get_lr(model_name: str, is_lora: bool = True) -> float:
    base_lr = 5e-05
    lora_multiplier = 10.0 if is_lora else 1.0

    lr = base_lr * lora_multiplier

    # Model-specific scaling based on hidden size
    if "llama" in model_name.lower():
        exponent_model = 0.781
    elif "qwen" in model_name.lower():
        exponent_model = 0.0775

    lr = lr * (2000 / _get_hidden_size(model_name)) ** exponent_model
    return lr
```

### LoRA Parameter Counting

```python
def get_lora_param_count(
    model_name: str,
    lora_rank: int = 32,
    include_experts: bool = True,
    shared_expert_outer_loras: bool = True,
) -> int:
    """Count trainable parameters in LoRA adapter."""
    dim_sum = 0

    for name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        if len(shape) == 2 and name.endswith(".weight"):
            if not any([v in name.split(".") for v in ["gate", "embed_tokens", ...]]):
                dim_sum += shape[0] + shape[1]

    return lora_rank * dim_sum
```

---

## Optimizer Configurations

### Adam Parameters

```python
adam_params = tinker.AdamParams(
    learning_rate=current_lr,
    beta1=0.9,      # Momentum
    beta2=0.95,     # RMSprop-like (NOT 0.999!)
    eps=1e-8        # Numerical stability
)
```

**Note:** The `beta2=0.95` default differs from typical Adam implementations (0.999). This is intentional for LLM training stability.

### Per-Step LR Adjustment

```python
# In training loop
learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
    lr_schedule=config.lr_schedule,
    step=step,
    total_steps=total_steps,
)
adam_params = tinker.AdamParams(learning_rate=learning_rate, ...)
```

---

## Learning Rate Scheduling

**File:** `tinker_cookbook/utils/lr_scheduling.py`

```python
LRSchedule = Literal["linear", "cosine", "constant"]

def compute_schedule_lr_multiplier(lr_schedule: LRSchedule, step: int, total_steps: int) -> float:
    if lr_schedule == "linear":
        return 1 - step / total_steps           # Linear decay to 0
    elif lr_schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))  # Cosine annealing
    elif lr_schedule == "constant":
        return 1                                 # No decay
```

### Usage

```python
@chz.chz
class Config:
    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "linear"
```

---

## Loss Functions

### Built-in Loss Functions

| Loss Function | Use Case | Key Inputs |
|--------------|----------|------------|
| `cross_entropy` | Supervised learning | `target_tokens`, `weights` |
| `importance_sampling` | Off-policy RL | `target_tokens`, `logprobs`, `advantages` |
| `ppo` | Clipped policy gradient | Same + clip thresholds |
| `cispo` | Clipped IS coefficient | Same as PPO |
| `dro` | Direct Reward Optimization | Same + `beta` |

### Importance Sampling (REINFORCE)

```python
# Corrects for sampling distribution mismatch
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```

### PPO Clipping

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
clipped_ratio = torch.clamp(prob_ratio, 0.8, 1.2)
unclipped_objective = prob_ratio * advantages
clipped_objective = clipped_ratio * advantages
loss = -torch.min(unclipped_objective, clipped_objective).sum()
```

### Custom Loss Functions

```python
def custom_loss_fn(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    loss = (logprobs ** 2).sum()
    return loss, {"custom_metric": loss.item()}

backward_result = training_client.forward_backward_custom(data, custom_loss_fn)
```

---

## Data Processing Abstractions

### Datum Structure

```python
# Core training data unit
tinker.Datum(
    model_input=tinker.ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "weights": TensorData.from_torch(weights),           # SL
        "logprobs": TensorData.from_torch(sampling_logprobs), # RL
        "advantages": TensorData.from_torch(advantages),      # RL
        "mask": TensorData.from_torch(mask),                  # RL
    }
)
```

### Trajectory to Datum Conversion

**File:** `tinker_cookbook/rl/data_processing.py`

```python
def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """Convert trajectory to training datums.

    Merges consecutive observations if they form extensions.
    Creates new Datum when sequence resets.
    """
    for transition in traj.transitions:
        ob = transition.ob
        ac_with_logprobs = transition.ac

        if _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            # Extension: merge with previous
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence):]
        else:
            # New sequence: emit current, start fresh
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat

        # Accumulate tokens, logprobs, advantages, mask
        SequenceAccumulator.full_sequence.extend(delta_ob_flat + ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend([0] * delta_ob_len + ac_with_logprobs.logprobs)
        SequenceAccumulator.advantages.extend([0] * delta_ob_len + [traj_advantage] * len(ac_tokens))
        SequenceAccumulator.mask.extend([0] * delta_ob_len + [1] * len(ac_tokens))
```

### Advantage Computation

```python
def compute_advantages(trajectory_groups_P: List[TrajectoryGroup]) -> List[torch.Tensor]:
    """Compute advantages centered within groups (GRPO-style)."""
    advantages_P = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        advantages_G = rewards_G - rewards_G.mean()  # Center within group
        advantages_P.append(advantages_G)

    return advantages_P
```

---

## Mapping to Elixir Ecosystem

### crucible_kitchen (Training Orchestration)

| Python Component | Elixir Module | Purpose |
|-----------------|---------------|---------|
| `supervised/train.py` | `CrucibleKitchen.SL.Runner` | SL training loop |
| `rl/train.py` | `CrucibleKitchen.RL.Runner` | RL training loop |
| `preference/train_dpo.py` | `CrucibleKitchen.DPO.Runner` | DPO training |
| `distillation/train_on_policy.py` | `CrucibleKitchen.Distillation.Runner` | Distillation |
| `checkpoint_utils.py` | `CrucibleKitchen.Checkpoint` | Save/resume |
| `utils/lr_scheduling.py` | `CrucibleKitchen.Scheduler` | LR schedules |
| `hyperparam_utils.py` | `CrucibleKitchen.Hyperparams` | LR/param scaling |

### tinkex (API Client)

| Python Component | Elixir Module | Status |
|-----------------|---------------|--------|
| `tinker.ServiceClient` | `Tinkex.ServiceClient` | Implemented |
| `tinker.TrainingClient` | `Tinkex.TrainingClient` | Implemented |
| `tinker.SamplingClient` | `Tinkex.SamplingClient` | Implemented |
| `tinker.AdamParams` | `Tinkex.Types.AdamParams` | Implemented |
| `tinker.Datum` | `Tinkex.Types.Datum` | Implemented |
| `tinker.ModelInput` | `Tinkex.Types.ModelInput` | Implemented |
| `tinker.TensorData` | `Tinkex.Types.TensorData` | Implemented |
| Loss functions | `Tinkex.Types.LossFnType` | Implemented |

### crucible_train (High-Level Training)

**Proposed modules:**

| Module | Purpose |
|--------|---------|
| `CrucibleTrain.Config` | Training configuration schemas |
| `CrucibleTrain.Pipeline` | End-to-end training pipelines |
| `CrucibleTrain.Experiment` | Experiment tracking integration |

---

## Implementation Status

### Already Implemented in tinkex

- `TrainingClient` GenServer with forward_backward, optim_step
- `SamplingClient` for inference
- `ServiceClient` for session/model creation
- Type structs: `AdamParams`, `Datum`, `ModelInput`, `TensorData`
- Future handling with polling
- Request chunking for large batches

### Missing in tinkex

- [ ] `load_state_with_optimizer` (resumption with optimizer state)
- [ ] `create_training_client_from_state_with_optimizer`
- [ ] `save_state` (full state save)
- [ ] `forward_backward_custom` (arbitrary loss functions)
- [ ] `compute_logprobs` (for DPO reference model)
- [ ] Loss function configuration (`loss_fn_config`)

### Not Yet in crucible_kitchen

- [ ] SL training loop runner
- [ ] RL training loop runner
- [ ] DPO training loop runner
- [ ] Checkpoint management
- [ ] LR scheduling
- [ ] Dataset abstractions
- [ ] Metrics logging (WandB, Neptune, Trackio)
- [ ] Environment abstractions for RL

---

## Recommendations

### 1. Priority Implementation Order

1. **Checkpoint resumption in tinkex** - Critical for long training runs
2. **SL training loop in crucible_kitchen** - Foundation for other paradigms
3. **LR scheduling** - Simple module, high impact
4. **Dataset abstractions** - Required for data loading
5. **RL training loop** - More complex, builds on SL
6. **DPO training** - Preference learning support

### 2. Elixir-Specific Patterns

**Use GenServers for:**
- Training loop state management
- Checkpoint coordination
- Metrics aggregation

**Use Tasks for:**
- Parallel rollout collection (like asyncio.gather)
- Async evaluation runs

**Use Broadway/GenStage for:**
- Streaming minibatch processing
- Async off-policy training queues

### 3. Config Schema Approach

Use Ecto.Changeset or custom validators:

```elixir
defmodule CrucibleKitchen.SL.Config do
  use Ecto.Schema
  import Ecto.Changeset

  embedded_schema do
    field :model_name, :string
    field :learning_rate, :float, default: 1.0e-4
    field :lr_schedule, Ecto.Enum, values: [:linear, :cosine, :constant]
    field :num_epochs, :integer, default: 1
    field :lora_rank, :integer, default: 32
    field :save_every, :integer, default: 20
    field :eval_every, :integer, default: 10
  end
end
```

### 4. Telemetry Integration

Leverage crucible_telemetry for metrics:

```elixir
:telemetry.execute(
  [:crucible_kitchen, :training, :step],
  %{loss: loss, learning_rate: lr},
  %{step: step, epoch: epoch}
)
```

---

## Appendix: Key Code References

### Python tinker-cookbook

- **SL Training:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/train.py`
- **RL Training:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/rl/train.py`
- **DPO Training:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/preference/train_dpo.py`
- **Checkpoint Utils:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/checkpoint_utils.py`
- **LR Scheduling:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/utils/lr_scheduling.py`
- **Hyperparams:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/hyperparam_utils.py`
- **Loss Functions:** `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/docs/losses.mdx`

### Elixir tinkex

- **Training Client:** `/home/home/p/g/North-Shore-AI/cns_ui/deps/tinkex/lib/tinkex/training_client.ex`
- **Adam Params:** `/home/home/p/g/North-Shore-AI/cns_ui/deps/tinkex/lib/tinkex/types/adam_params.ex`
- **Service Client:** `/home/home/p/g/North-Shore-AI/cns_ui/deps/tinkex/lib/tinkex/service_client.ex`
