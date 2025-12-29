# Tinker-Cookbook Core Library Analysis

**Date:** 2025-12-28
**Purpose:** Comprehensive analysis of tinker-cookbook core library for Elixir port planning
**Target Destination:** crucible_kitchen (abstraction layer) + other ecosystem projects

---

## Executive Summary

The tinker-cookbook library is a comprehensive ML training toolkit built around the Tinker API for distributed training. It provides:

1. **Chat/Message Rendering** - Token-level formatting for different model families
2. **Supervised Fine-Tuning (SFT)** - Complete training loops with checkpointing
3. **Reinforcement Learning** - MDP-based RL with rollouts, trajectories, advantage computation
4. **Direct Preference Optimization (DPO)** - Preference learning from comparisons
5. **On-Policy Distillation** - Teacher-student knowledge transfer
6. **Evaluation Framework** - Extensible evaluators including Inspect AI integration
7. **Utilities** - Logging, tracing, hyperparameter estimation, checkpoint management

---

## Module-by-Module Analysis

### 1. Core Abstractions (`completers.py`, `renderers.py`)

#### 1.1 Completers (`completers.py`)

**Purpose:** Define interfaces for model/policy completion at different abstraction levels.

**Key Abstractions:**
- `TokenCompleter` - Base interface operating on token sequences
- `MessageCompleter` - Higher-level interface operating on messages
- `TinkerTokenCompleter` - Implementation using Tinker sampling client
- `TinkerMessageCompleter` - Implementation wrapping token completer with renderer
- `TokensWithLogprobs` - Data class holding tokens and their log probabilities

**Implementation Details:**
```python
StopCondition: TypeAlias = list[str] | list[int]

@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Core abstraction needed by all training paradigms. Maps well to Elixir behaviours.

---

#### 1.2 Renderers (`renderers.py`)

**Purpose:** Convert between chat messages and token sequences for different model families.

**Key Types:**
- `Message` - TypedDict with role, content, optional tool_calls and thinking
- `TrainOnWhat` - Enum controlling which tokens get loss weights
- `Renderer` - Base class for all renderers

**Supported Renderers:**
| Renderer | Target Model Family | Features |
|----------|-------------------|----------|
| `RoleColonRenderer` | Generic/DeepSeek | Simple `User:` / `Assistant:` format |
| `Llama3Renderer` | Meta Llama 3.x | Special tokens `<|start_header_id|>`, `<|eot_id|>` |
| `Qwen3Renderer` | Qwen 3.x | `<|im_start|>`, `<|im_end|>`, `<think>` support |
| `Qwen3DisableThinkingRenderer` | Qwen 3.x | Thinking disabled variant |
| `Qwen3InstructRenderer` | Qwen 3.x Instruct 2507 | No thinking tags |
| `DeepSeekV3Renderer` | DeepSeek V3 | Special tokens, thinking support |
| `DeepSeekV3DisableThinkingRenderer` | DeepSeek V3 | Thinking disabled |
| `GptOssRenderer` | GPT-OSS | Channels (analysis, commentary, final) |

**Core Methods:**
- `build_supervised_example(messages, train_on_what)` -> tokens, weights
- `build_generation_prompt(messages, role, prefill)` -> ModelInput
- `get_stop_sequences()` -> stop tokens
- `parse_response(response)` -> Message, success_bool

**TrainOnWhat Options:**
- `LAST_ASSISTANT_MESSAGE` - Only final assistant turn
- `ALL_ASSISTANT_MESSAGES` - All assistant turns
- `ALL_MESSAGES` - All message content
- `ALL_TOKENS` - Everything including headers
- `ALL_USER_AND_SYSTEM_MESSAGES` - Non-assistant content

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Essential for any SFT/RL work. Consider using protocols for extensibility.

---

### 2. Supervised Learning (`supervised/`)

#### 2.1 Types (`supervised/types.py`)

**Key Abstractions:**
- `SupervisedDataset` - Base class for datasets providing batches of `tinker.Datum`
- `SupervisedDatasetBuilder` - Factory pattern for dataset construction
- `ChatDatasetBuilder` - Builder specifically for chat-formatted data

**Interface:**
```python
class SupervisedDataset:
    def get_batch(self, index: int) -> list[tinker.Datum]
    def __len__(self) -> int
    def set_epoch(self, seed: int = 0)  # For shuffling
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Dataset abstraction is fundamental. Use Elixir Stream for lazy loading.

---

#### 2.2 Data Processing (`supervised/data.py`)

**Key Implementations:**
- `SupervisedDatasetFromHFDataset` - Wraps HuggingFace datasets
- `StreamingSupervisedDatasetFromHFDataset` - Streaming variant for large datasets
- `FromConversationFileBuilder` - Loads JSONL conversation files

**Helper Functions:**
- `conversation_to_datum()` - Converts messages to training datum
- `datum_from_tokens_weights()` - Creates Datum from raw tokens/weights

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Data loading utilities. May delegate to `crucible_datasets` for actual loaders.

---

#### 2.3 Training Loop (`supervised/train.py`)

**Purpose:** Complete SFT training loop with pipelining for GPU efficiency.

**Config Parameters:**
```python
class Config:
    log_path: str
    model_name: str
    load_checkpoint_path: str | None
    dataset_builder: SupervisedDatasetBuilder
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"  # or other schedules
    num_epochs: int = 1
    lora_rank: int = 32
    evaluator_builders: list[EvaluatorBuilder]
    save_every: int = 20
    eval_every: int = 10
    adam_beta1/beta2/eps: float  # Optimizer params
    wandb_project/name: str | None
```

**Training Flow:**
1. Resume from checkpoint if available
2. Create training client (LoRA or full)
3. Build dataset and evaluators
4. Pipeline: submit_batch -> finish_batch (overlap compute)
5. Periodic checkpointing and evaluation
6. Final checkpoint

**Elixir Port Location:** `crucible_kitchen` (orchestration) + `tinkex` (API calls)
**Priority:** HIGH
**Notes:** Core training loop. Consider GenServer for state management.

---

#### 2.4 NLL Evaluator (`supervised/nll_evaluator.py`)

**Purpose:** Compute negative log-likelihood on test set during training.

**Implementation:**
```python
class NLLEvaluator(TrainingClientEvaluator):
    def __call__(self, training_client) -> dict[str, float]:
        # Forward pass, compute weighted NLL
        return {"nll": nll}
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Standard evaluation metric.

---

#### 2.5 Common Utilities (`supervised/common.py`)

**Key Functions:**
- `compute_mean_nll(logprobs_list, weights_list)` - Weighted NLL computation
- `datum_from_tokens_weights(tokens, weights, max_length)` - Datum construction

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Utility functions, can be in a helpers module.

---

### 3. Reinforcement Learning (`rl/`)

#### 3.1 Types (`rl/types.py`)

**Core Abstractions:**

```python
# Type aliases
Action = list[int]
Observation = tinker.ModelInput
Logprobs = list[float]
Metrics = dict[str, float | int]

# Data classes
@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics

@dataclass
class Transition:
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics

@dataclass
class Trajectory:
    transitions: list[Transition]
    final_ob: Observation

@dataclass
class TrajectoryGroup:
    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]
    metrics_G: list[Metrics]
```

**Abstract Classes:**
- `Env` - Stateful environment (single episode)
  - `initial_observation()` -> (Observation, StopCondition)
  - `step(action)` -> StepResult
- `EnvGroupBuilder` - Creates groups of environments for GRPO-style algorithms
  - `make_envs()` -> Sequence[Env]
  - `compute_group_rewards(trajectories, envs)` -> list[(float, Metrics)]
  - `logging_tags()` -> list[str]
- `RLDataset` - Produces batches of EnvGroupBuilders
- `RLDatasetBuilder` - Factory for RLDataset

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Core RL abstractions. Consider using protocols and GenServers for stateful envs.

---

#### 3.2 Rollouts (`rl/rollouts.py`)

**Purpose:** Execute policy in environments to collect trajectories.

**Key Functions:**
```python
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory
async def do_group_rollout(env_group_builder: EnvGroupBuilder, policy: TokenCompleter) -> TrajectoryGroup
```

**Flow:**
1. Get initial observation from env
2. Loop until episode done:
   - Policy samples action
   - Env steps with action
   - Record transition
3. Return complete trajectory

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Async nature maps well to Elixir Tasks/async.

---

#### 3.3 Data Processing (`rl/data_processing.py`)

**Key Functions:**
- `compute_advantages(trajectory_groups)` - Center rewards within groups (GRPO-style)
- `trajectory_to_data(traj, advantage)` - Convert trajectory to training Datum
- `assemble_training_data(groups, advantages)` -> (data, metadata)
- `remove_constant_reward_groups(groups)` - Filter groups with no variance

**Advantage Computation:**
```python
advantages_G = rewards_G - rewards_G.mean()  # Per-group centering
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Numerical operations may need Nx integration.

---

#### 3.4 Metrics (`rl/metrics.py`)

**Key Functions:**
- `compute_kl_sample_train(data, training_logprobs)` - KL divergence between sampling and training policies
- `compute_post_kl(data, post_sampling_client)` - Post-update KL
- `incorporate_kl_penalty(data, base_client, coef, discount)` - Add KL penalty to advantages
- `discounted_future_sum_vectorized(x, gamma)` - Efficient discounted returns
- `compute_sampling_client_metrics(wrapped_groups)` - Staleness metrics for async training

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Numerical operations, consider Nx for tensor ops.

---

#### 3.5 Problem Environment (`rl/problem_env.py`)

**Purpose:** Base class for question-answering style RL environments.

**Abstract Methods:**
- `get_question()` -> str
- `check_answer(sample_str)` -> bool
- `check_format(sample_str)` -> bool
- `get_reference_answer()` -> str

**Reward Structure:**
```python
total_reward = format_coef * (correct_format - 1) + correct_answer
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Extensible base for domain-specific environments.

---

#### 3.6 RL Training Loop (`rl/train.py`)

**Purpose:** Complete RL training with multiple training modes.

**Training Modes:**
1. `do_sync_training` - Fully synchronous on-policy
2. `do_sync_training_with_stream_minibatch` - Synchronous with minibatch streaming
3. `do_async_training` - Asynchronous off-policy with staleness limits

**Config Parameters:**
```python
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder
    model_name: str
    max_tokens: int
    loss_fn: Literal["importance_sampling", "ppo"]
    kl_penalty_coef: float
    kl_discount_factor: float
    num_substeps: int  # Gradient accumulation
    async_config: AsyncConfig | None  # For async mode
    stream_minibatch_config: StreamMinibatchConfig | None
```

**AsyncConfig:**
```python
class AsyncConfig:
    max_steps_off_policy: int  # Discard stale samples
    groups_per_batch: int  # Minimum batch size
```

**StreamMinibatchConfig:**
```python
class StreamMinibatchConfig:
    groups_per_batch: int
    num_minibatches: int  # For gradient accumulation overlap
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Complex state management - consider GenStateMachine.

---

#### 3.7 Metric Utilities (`rl/metric_util.py`)

**Key Classes:**
- `RLTestSetEvaluator` - Evaluates policy on test set

**Key Functions:**
- `compute_trajectory_metrics(groups, tags)` - Aggregated metrics by env tags
- `dataset_to_env_group_builders(dataset)` - Flatten dataset

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Evaluation utilities.

---

#### 3.8 Preference Environments (`rl/preference_envs.py`)

**Purpose:** RL environments for pairwise preference learning.

**Key Classes:**
- `PreferenceEnv` - Single-step env for preference-based RL
- `PairwisePreferenceGroupBuilder` - Runs tournament-style comparisons
- `PairwisePreferenceDataset` - Dataset of preference comparisons
- `PairwisePreferenceRLDatasetBuilder` - Factory for preference datasets

**Tournament Patterns:**
- `ALL_PAIRS_BOTH_WAYS` - Full N*(N-1) comparisons
- `ALL_PAIRS_ONE_WAY` - N*(N-1)/2 comparisons

**Reward Computation:**
```python
# Win/loss delta computed from pairwise preference model
reward = win_minus_loss / matchup_count + format_coef * (is_valid - 1)
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Specialized for RLHF with preference models.

---

### 4. Preference Learning / DPO (`preference/`)

#### 4.1 Types (`preference/types.py`)

**Core Data Structures:**
```python
@dataclass
class Comparison:
    prompt_conversation: list[Message]
    completion_A: list[Message]
    completion_B: list[Message]

    def swap(self) -> Comparison  # For data augmentation

@dataclass
class LabeledComparison:
    comparison: Comparison
    label: Literal["A", "B", "Tie"]

    def swap(self) -> LabeledComparison
```

**Key Classes:**
- `ComparisonRenderer` - Renders comparisons for preference model
- `PreferenceModel` - Returns preference score (-1 to 1) for comparison
- `PreferenceModelFromChatRenderer` - Uses sampling to get preference

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Types for preference-based training.

---

#### 4.2 Preference Datasets (`preference/preference_datasets.py`)

**Key Classes:**
- `ComparisonDatasetBuilder` - Base class for loading comparison datasets
- `ChatDatasetBuilderFromComparisons` - Converts comparisons to SFT format
- `ComparisonBuilderFromJsonl` - Loads from JSONL files

**Elixir Port Location:** `crucible_kitchen` + `crucible_datasets`
**Priority:** MEDIUM
**Notes:** Dataset utilities for preference data.

---

#### 4.3 DPO Datasets (`preference/dpo_datasets.py`)

**Purpose:** Build datasets specifically for DPO training (chosen/rejected pairs).

**Key Classes:**
- `DPODatasetBuilderFromComparisons` - Creates paired data for DPO loss

**Output Format:** Alternating chosen/rejected Datum pairs.

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Specific to DPO training paradigm.

---

#### 4.4 DPO Training (`preference/train_dpo.py`)

**Purpose:** Direct Preference Optimization training loop.

**DPO Loss:**
```python
def compute_dpo_loss(chosen_logprobs, rejected_logprobs,
                     chosen_ref_logprobs, rejected_ref_logprobs, beta):
    chosen_log_ratio = chosen_logprobs - chosen_ref_logprobs
    rejected_log_ratio = rejected_logprobs - rejected_ref_logprobs
    losses = -log(sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)))
    return losses.mean()
```

**Key Metrics:**
- `dpo_loss` - The DPO objective
- `accuracy` - Fraction where policy prefers chosen
- `margin` - Average reward difference
- `chosen_reward` / `rejected_reward` - Per-class rewards

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Core DPO implementation.

---

### 5. Distillation (`distillation/`)

#### 5.1 Datasets (`distillation/datasets.py`)

**Purpose:** Dataset utilities for on-policy distillation.

**Key Classes:**
- `TeacherConfig` - Configuration for teacher model
- `DistillationDatasetConfig` - Combines dataset, teacher, and batch config
- `CompositeDataset` - Combines multiple datasets with different teachers
- `PromptOnlyEnv` - Environment providing only prompts (no rewards)
- `PromptOnlyDataset` - Dataset of prompts
- `PromptOnlyDatasetBuilder` - Factory for prompt datasets

**Supported Datasets:**
- DeepMath (from HuggingFace)
- Tulu3 (from HuggingFace)

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Distillation-specific dataset handling.

---

#### 5.2 On-Policy Distillation Training (`distillation/train_on_policy.py`)

**Purpose:** Training loop for on-policy distillation with KL penalty.

**Key Concept:** Uses reverse KL (student vs teacher) as the training signal:
```python
reverse_kl = log_p_student - log_p_teacher
advantage = -kl_penalty_coef * reverse_kl
```

**Multi-Teacher Support:** Can distill from multiple teachers simultaneously with per-dataset metrics.

**Config:**
```python
class Config:
    dataset_configs: List[DistillationDatasetConfig]  # Multi-teacher
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    # ... standard RL config params
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Specialized training paradigm.

---

### 6. Evaluation (`eval/`)

#### 6.1 Evaluator Base Classes (`eval/evaluators.py`)

**Core Interfaces:**
```python
class TrainingClientEvaluator:
    async def __call__(self, training_client) -> dict[str, float]

class SamplingClientEvaluator:
    async def __call__(self, sampling_client) -> dict[str, float]
```

**Type Aliases:**
```python
EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** HIGH
**Notes:** Core evaluation abstraction. Use behaviours.

---

#### 6.2 Custom Evaluators (`eval/custom_evaluators.py`)

**Purpose:** Template for custom evaluation tasks.

**Example Implementation:**
```python
class CustomEvaluator(SamplingClientEvaluator):
    def __init__(self, dataset, grader_fn, model_name, renderer_name):
        self.dataset = dataset
        self.grader_fn = grader_fn  # (response, target) -> bool
        # ...

    async def __call__(self, sampling_client):
        # Run inference, compute accuracy
        return {"accuracy": num_correct / num_examples}
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** LOW
**Notes:** Example code, mainly for reference.

---

#### 6.3 Inspect AI Integration (`eval/inspect_evaluators.py`, `eval/inspect_utils.py`)

**Purpose:** Integrate with Inspect AI evaluation framework.

**Key Classes:**
- `InspectEvaluatorBuilder` - Configuration for Inspect tasks
- `InspectEvaluator` - Wraps Inspect AI evaluation
- `InspectAPIFromTinkerSampling` - Adapts Tinker sampling to Inspect API

**Features:**
- Run arbitrary Inspect tasks
- Configure temperature, max_tokens, etc.
- Extract metrics from Inspect results

**Elixir Port Location:** Likely not needed (Inspect is Python-specific)
**Priority:** LOW
**Notes:** Consider alternative eval frameworks for Elixir.

---

### 7. Utilities (`utils/`)

#### 7.1 Logging Tree (`utils/logtree.py`, `utils/logtree_formatters.py`)

**Purpose:** Hierarchical structured logging with HTML output.

**Key Functions:**
```python
def init_trace(title: str, path: str | None = None)
def scope_header(name: str)  # Context manager for nested logging
def scope_header_decorator(func)  # Decorator version
def log_text(text: str)
def log_formatter(formatter: LogtreeFormatter)
def table(rows: list[dict], caption: str | None = None)
```

**Formatters:**
- `ConversationFormatter` - Pretty-print chat messages
- `CodeFormatter` - Syntax-highlighted code
- `TableFormatter` - Tabular data

**Elixir Port Location:** `crucible_telemetry` or new `crucible_logging`
**Priority:** MEDIUM
**Notes:** Nice-to-have for debugging. Consider integration with Logger.

---

#### 7.2 Tracing (`utils/trace.py`)

**Purpose:** Performance tracing for async operations.

**Features:**
- Trace events in Chrome Tracing format
- Track async task creation/completion
- Nested scope tracking

**Key Functions:**
```python
def trace_init(output_file: str)
def scope(func)  # Decorator
def get_scope_context() -> ScopeContext
```

**Output:** JSONL file compatible with chrome://tracing or Perfetto.

**Elixir Port Location:** `crucible_telemetry`
**Priority:** LOW
**Notes:** Elixir has built-in tracing via :dbg and Telemetry.

---

#### 7.3 Miscellaneous Utilities (`utils/misc_utils.py`)

**Utility Functions:**
- `safezip(*args)` - Zip with strict length checking
- `split_list(lst, n)` - Split list into n chunks
- `timed(name, metrics)` - Context manager for timing
- `all_same(values)` - Check if all values are equal
- `dict_mean(dicts)` - Average values across dicts
- `not_none(x)` - Assert value is not None

**Elixir Port Location:** `crucible_kitchen`
**Priority:** LOW
**Notes:** Simple utilities, Elixir has equivalents.

---

#### 7.4 File Utilities (`utils/file_utils.py`)

**Purpose:** File handling with blobfile support.

**Key Functions:**
```python
def ensure_path_exists(path: str)
def read_json(path: str) -> dict
def write_json(path: str, data: dict)
def list_files(path: str, pattern: str) -> list[str]
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** LOW
**Notes:** Elixir File module covers most needs.

---

#### 7.5 Learning Rate Scheduling (`utils/lr_scheduling.py`)

**Purpose:** Compute learning rate multipliers for different schedules.

**Supported Schedules:**
- `constant` - No decay
- `linear` - Linear decay to 0
- `cosine` - Cosine annealing
- `wsd` (warmup-stable-decay) - Complex schedule with warmup

**Key Function:**
```python
def compute_schedule_lr_multiplier(lr_schedule: str, step: int, total_steps: int) -> float
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** MEDIUM
**Notes:** Simple math, easy to port.

---

#### 7.6 Code State (`utils/code_state.py`)

**Purpose:** Capture git state and code snapshot for reproducibility.

**Key Functions:**
```python
def get_git_info() -> dict  # Commit hash, branch, status
def save_code_snapshot(path: str)  # Archive current code
```

**Elixir Port Location:** `crucible_telemetry` or `crucible_harness`
**Priority:** LOW
**Notes:** Useful for experiment tracking.

---

#### 7.7 ML Logging (`utils/ml_log.py`)

**Purpose:** Unified logging to multiple backends (file, WandB).

**Key Classes:**
```python
class Logger:
    def log_metrics(self, metrics: dict, step: int)
    def get_logger_url(self) -> str | None
    def close()

def setup_logging(log_dir, wandb_project, config, ...) -> Logger
```

**Backends:**
- JSONL file logging
- Weights & Biases integration

**Elixir Port Location:** `crucible_telemetry`
**Priority:** MEDIUM
**Notes:** Consider Telemetry + custom reporters.

---

#### 7.8 Colorized Formatting (`utils/format_colorized.py`)

**Purpose:** Terminal-friendly visualization of token sequences with weights.

**Key Function:**
```python
def format_colorized(tokens: list[int], weights: list[float], tokenizer) -> str
```

**Elixir Port Location:** `crucible_kitchen` or display utility
**Priority:** LOW
**Notes:** Nice for debugging, but low priority.

---

### 8. Other Top-Level Modules

#### 8.1 Model Info (`model_info.py`)

**Purpose:** Metadata about supported models.

**Key Data:**
```python
@dataclass
class ModelAttributes:
    organization: str  # meta-llama, Qwen, etc.
    version_str: str
    size_str: str
    is_chat: bool
    is_vl: bool = False  # Vision-language
```

**Key Functions:**
- `get_model_attributes(model_name)` - Get metadata
- `get_recommended_renderer_name(model_name)` - Get renderer for model
- `get_recommended_renderer_names(model_name)` - List of compatible renderers

**Elixir Port Location:** `crucible_kitchen` or `tinkex`
**Priority:** MEDIUM
**Notes:** Configuration data, easy to port.

---

#### 8.2 Hyperparameter Utilities (`hyperparam_utils.py`)

**Purpose:** Estimate good hyperparameters for fine-tuning.

**Key Functions:**
- `get_lora_param_count(model_name, lora_rank)` - Estimate LoRA parameter count
- `get_lr(model_name, is_lora)` - Get recommended learning rate
- `get_lora_lr_multiplier(model_name)` - LR scaling factor
- `get_full_finetune_param_count(model_name)` - Total parameters

**Implementation:** Reads safetensors headers via HTTP range requests for efficiency.

**Elixir Port Location:** `crucible_kitchen`
**Priority:** LOW
**Notes:** Heuristic utilities, low priority.

---

#### 8.3 Display (`display.py`)

**Purpose:** Visualization utilities for training examples.

**Key Function:**
```python
def colorize_example(datum, tokenizer, key="weights") -> str
```

**Elixir Port Location:** `crucible_kitchen`
**Priority:** LOW
**Notes:** Debugging utility.

---

#### 8.4 CLI Utilities (`cli_utils.py`)

**Purpose:** Command-line argument handling with chz.

**Elixir Port Location:** Not needed
**Priority:** SKIP
**Notes:** Elixir has different CLI patterns.

---

#### 8.5 Checkpoint Utilities (`checkpoint_utils.py`)

**Purpose:** Save and load training checkpoints.

**Key Functions:**
```python
def save_checkpoint(training_client, name, log_path, loop_state, kind) -> dict
def save_checkpoint_async(...)  # Async version
def get_last_checkpoint(log_path) -> dict | None
```

**Checkpoint Types:**
- `sampler` - Weights for inference
- `training` - Full training state (optimizer, etc.)
- `both` - Both types

**Elixir Port Location:** `crucible_kitchen` + `tinkex`
**Priority:** HIGH
**Notes:** Essential for resumable training.

---

#### 8.6 Tokenizer Utilities (`tokenizer_utils.py`)

**Purpose:** Wrapper around HuggingFace tokenizers.

**Key Functions:**
```python
def get_tokenizer(model_name: str) -> Tokenizer
```

**Tokenizer Interface:**
- `encode(text, add_special_tokens)` -> list[int]
- `decode(tokens)` -> str
- `bos_token`, `eos_token` properties

**Elixir Port Location:** `tinkex`
**Priority:** HIGH
**Notes:** Already exists in tinkex.

---

## Architecture Mapping

### crucible_kitchen (Core Abstraction Layer)

**Should Include:**
1. **Completers** - TokenCompleter, MessageCompleter behaviours
2. **Renderers** - All renderer implementations
3. **Dataset Abstractions** - SupervisedDataset, RLDataset protocols
4. **RL Types** - Env, Trajectory, TrajectoryGroup, etc.
5. **Training Orchestration** - SFT, RL, DPO training loops
6. **Evaluator Framework** - Base behaviours for evaluators
7. **Checkpoint Management** - Save/load utilities
8. **LR Scheduling** - Schedule computation functions

**Estimated Modules:**
- `CrucibleKitchen.Completer` - Behaviours and implementations
- `CrucibleKitchen.Renderer` - All renderers
- `CrucibleKitchen.Dataset` - Dataset protocols
- `CrucibleKitchen.RL` - RL-specific types and utilities
- `CrucibleKitchen.Training` - Training loop GenServers
- `CrucibleKitchen.Evaluator` - Evaluation framework
- `CrucibleKitchen.Checkpoint` - Checkpoint management

### tinkex (Tinker API Client)

**Already Has:**
- Tokenizer integration
- API client

**Should Add:**
- Training client abstraction (LoRA, full fine-tune)
- Sampling client abstraction
- Datum/ModelInput types

### crucible_telemetry

**Should Add:**
- ML metrics logging (file + WandB-like)
- Performance tracing integration
- Code state capture

### crucible_datasets

**Should Add:**
- HuggingFace dataset loaders for:
  - DeepMath
  - Tulu3
  - Preference datasets
- Streaming dataset support

---

## Priority Summary

### HIGH Priority
1. Completers (TokenCompleter, MessageCompleter)
2. Renderers (Llama3, Qwen3, DeepSeekV3, etc.)
3. Dataset abstractions (SupervisedDataset, RLDataset)
4. RL types (Env, Trajectory, TrajectoryGroup, etc.)
5. SFT training loop
6. RL training loop
7. DPO training
8. Checkpoint utilities
9. Evaluator base classes

### MEDIUM Priority
1. Data processing (conversation_to_datum, etc.)
2. NLL evaluator
3. Advantage computation
4. KL metrics
5. Problem environments
6. Preference environments
7. LR scheduling
8. Model info
9. ML logging
10. Distillation training

### LOW Priority
1. Logtree logging
2. Tracing
3. File utilities
4. Hyperparameter estimation
5. Display/colorization utilities
6. Code state capture
7. Inspect AI integration (likely skip for Elixir)

---

## Implementation Notes

### Elixir-Specific Considerations

1. **Async Operations**: Python's asyncio maps well to Elixir Tasks and async/await patterns.

2. **State Management**: Python classes with mutable state should become GenServers or Agents.

3. **Type System**: TypedDicts and dataclasses map to Elixir structs with @type specs.

4. **Protocols vs Behaviours**:
   - Use behaviours for abstract base classes (Renderer, Evaluator)
   - Use protocols for duck typing (Dataset access patterns)

5. **Numerical Operations**: Consider Nx for tensor operations currently using torch/numpy.

6. **Configuration**: Python's chz pattern maps to Elixir's compile-time config + runtime options.

### Testing Strategy

1. Port test cases alongside implementation
2. Use property-based testing for numerical code
3. Integration tests against actual Tinker API

### Dependencies

- `tinkex` - Required for API communication
- `nx` - For tensor operations
- `jason` - JSON handling
- `ex_doc` - Documentation

---

## Next Steps

1. Set up crucible_kitchen project structure
2. Define core behaviours (Completer, Renderer, Dataset, Evaluator)
3. Implement renderers (start with Llama3 as reference)
4. Port dataset abstractions
5. Implement SFT training loop as GenServer
6. Add RL training capabilities
7. Integrate with existing crucible_ ecosystem
