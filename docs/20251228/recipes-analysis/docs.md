# Tinker-Cookbook Recipes Analysis for Crucible Extraction

**Date:** 2025-12-28
**Purpose:** Comprehensive analysis of tinker-cookbook recipes to identify patterns and functionality for extraction into the Elixir crucible_ ecosystem.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Recipe Analysis](#recipe-analysis)
4. [Core Library Patterns](#core-library-patterns)
5. [Extractable Functionality](#extractable-functionality)
6. [Missing Abstractions](#missing-abstractions)
7. [Recommended Crucible Modules](#recommended-crucible-modules)

---

## Executive Summary

The tinker-cookbook library provides a comprehensive framework for fine-tuning language models using the Tinker service. The library is organized around three main paradigms:

1. **Supervised Learning (SL)** - Traditional fine-tuning on labeled data
2. **Reinforcement Learning (RL)** - Training with reward signals from environments
3. **Preference Learning** - DPO and RLHF for alignment

Key patterns that should be extracted to Elixir:

- **Training Loop Orchestration** - Pipelined async training with checkpointing
- **Environment Abstractions** - Env/EnvGroup for RL rollouts
- **Renderer System** - Model-specific chat template handling
- **Metrics & Logging** - Structured logging with multiple backends
- **Evaluation Framework** - Pluggable evaluators during training
- **Dataset Management** - Batching, shuffling, streaming from various sources
- **Grading/Verification** - Answer checking and format validation

---

## Architecture Overview

### Core Components

```
tinker-cookbook/
  tinker_cookbook/
    rl/
      types.py       # Env, EnvGroupBuilder, Trajectory, etc.
      train.py       # RL training loop
      rollouts.py    # do_single_rollout, do_group_rollout
      problem_env.py # ProblemEnv base class
    supervised/
      types.py       # SupervisedDataset, ChatDatasetBuilder
      train.py       # SL training loop
      data.py        # Dataset implementations
      common.py      # datum_from_model_input_weights
    preference/
      train_dpo.py   # DPO training loop
    renderers.py     # Model-specific chat templates
    completers.py    # TokenCompleter, MessageCompleter
    hyperparam_utils.py  # LR calculation
    checkpoint_utils.py  # Save/load training state
    eval/
      evaluators.py  # Evaluator interfaces
    utils/
      ml_log.py      # Multi-backend logging
      logtree.py     # Hierarchical HTML reports
```

### Builder Pattern

The library uses a builder pattern with `chz` dataclasses for configuration:

```python
@chz.chz
class Config:
    model_name: str
    learning_rate: float = 1e-4
    dataset_builder: SupervisedDatasetBuilder
    # ...
```

This allows serialization and CLI entrypoints.

---

## Recipe Analysis

### 1. sl_basic.py - Supervised Learning Basics

**Location:** `/recipes/sl_basic.py`

**Purpose:** Minimal example of supervised fine-tuning on a single batch.

**Core Library Features Used:**
- `tinker.ServiceClient` - API client creation
- `tinker.SamplingParams` - Sampling configuration
- `renderer.build_supervised_example()` - Convert messages to tokens/weights
- `datum_from_model_input_weights()` - Create training datum
- `forward_backward()` and `optim_step()` - Training step

**Recipe-Specific Code:**
- Inline data definition (should use dataset builders)
- Manual client setup (should use factory)

**Extractable Patterns:**
- Single-batch training step abstraction
- Data conversion pipeline

---

### 2. sl_loop.py - Supervised Learning Loop

**Location:** `/recipes/sl_loop.py`

**Purpose:** Minimal training loop with multiple batches and epochs.

**Core Library Features Used:**
- Async pipelining (`forward_backward_async`, `optim_step_async`)
- Dataset iteration with batching
- Learning rate scheduling

**Recipe-Specific Code:**
- Loss tracking and printing
- Simple epoch/batch loop

**Extractable Patterns:**
- Pipelined training loop (submit next batch while waiting for current)
- NLL computation from logprobs

---

### 3. rl_basic.py - Reinforcement Learning Basics

**Location:** `/recipes/rl_basic.py`

**Purpose:** Minimal RL training on a simple counting environment.

**Core Library Features Used:**
- `Env` protocol - Environment interface
- `do_single_rollout()` - Execute one episode
- `TinkerTokenCompleter` - Policy wrapper
- `importance_sampling` loss function

**Recipe-Specific Code:**
- `CountingEnv` - Simple test environment
- Manual trajectory to datum conversion

**Extractable Patterns:**
- Environment interface
- Rollout execution
- Trajectory processing for training

---

### 4. rl_loop.py - Reinforcement Learning Loop

**Location:** `/recipes/rl_loop.py`

**Purpose:** Full RL training loop with proper pipelining.

**Core Library Features Used:**
- `EnvGroupBuilder` - Create groups of environments
- `do_group_rollout()` - Execute rollouts for a group
- `trajectories_to_rl_data()` - Convert trajectories to training data
- Advantage centering within groups

**Recipe-Specific Code:**
- Environment group creation
- Metrics aggregation

**Extractable Patterns:**
- Group-based RL training
- Advantage normalization
- Policy update pipelining

---

### 5. math_rl/ - Math Problem RL Training

**Location:** `/recipes/math_rl/`

**Files:**
- `train.py` - Training loop
- `math_env.py` - GSM8K environment using ProblemEnv
- `arithmetic_env.py` - Simple arithmetic environment
- `math_grading.py` - Answer extraction and verification

**Purpose:** Train models on math reasoning tasks.

**Core Library Features Used:**
- `ProblemEnv` - Base class for single-turn Q&A problems
- `ProblemGroupBuilder` - Create problem groups
- `MathDataset` - Load GSM8K problems

**Recipe-Specific Code (Should Be Generalized):**
```python
def extract_answer(sample: str) -> str | None:
    """Extract boxed answer from response."""
    patterns = [
        r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}",
        r"the answer is:?\s*\$?(\d+(?:\.\d+)?)",
        # ...
    ]
```

**Extractable Patterns:**
- Answer extraction with multiple fallback patterns
- Numeric equivalence checking with tolerance
- Format validation (boxed answers, step markers)

---

### 6. code_rl/ - Code Generation RL Training

**Location:** `/recipes/code_rl/`

**Files:**
- `train.py` - Training loop
- `code_env.py` - LiveCodeBench environment
- `code_grading.py` - Code execution and test verification
- `lcb_utils.py` - LiveCodeBench dataset utilities

**Purpose:** Train models on coding tasks with execution feedback.

**Core Library Features Used:**
- `ProblemEnv` subclass for code problems
- Multi-turn conversation prefix support

**Recipe-Specific Code (Should Be Generalized):**
```python
def extract_code(sample: str) -> str | None:
    """Extract Python code from markdown blocks."""
    patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
    ]
```

```python
def execute_tests(code: str, tests: list[str]) -> tuple[bool, dict]:
    """Execute code against test cases in sandbox."""
```

**Extractable Patterns:**
- Code extraction from markdown
- Test execution sandboxing
- Pass@k evaluation metrics

---

### 7. distillation/ - Knowledge Distillation

**Location:** `/recipes/distillation/`

**Files:**
- `on_policy_distillation.py` - Distill from teacher on-policy
- `off_policy_reasoning.py` - Distill from pre-generated data
- `on_policy_multi_teacher.py` - Ensemble distillation

**Purpose:** Transfer knowledge from teacher models to students.

**Core Library Features Used:**
- Multiple `SamplingClient` instances for different models
- Conversation rendering for both teacher and student
- SL training with teacher-generated labels

**Recipe-Specific Code (Should Be Generalized):**
```python
async def get_teacher_responses(
    prompts: list[str],
    teacher_client: SamplingClient,
    n_samples: int = 1,
) -> list[str]:
    """Sample responses from teacher model."""
```

**Extractable Patterns:**
- Multi-model orchestration
- On-policy vs off-policy data generation
- Teacher selection/routing

---

### 8. preference/ - Preference Learning (DPO/RLHF)

**Location:** `/recipes/preference/`

**Subdirectories:**
- `dpo/train.py` - Direct Preference Optimization
- `shorter/` - Length-based preference training
- `rlhf/rlhf_pipeline.py` - Full RLHF pipeline
- `datasets.py` - Preference pair datasets

**Purpose:** Align models using human/AI preferences.

**Core Library Features Used (train_dpo.py):**
- Reference model for KL divergence
- `forward_backward_custom()` with DPO loss
- Paired data handling (chosen/rejected)

**Recipe-Specific Code (Should Be Generalized):**
```python
def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
```

**Extractable Patterns:**
- Preference pair data format
- DPO loss computation
- Reference model management
- Reward margin metrics

---

### 9. multiplayer_rl/ - Multi-Agent RL

**Location:** `/recipes/multiplayer_rl/`

**Subdirectories:**
- `guess_number/` - Number guessing game (hinter + guesser)
- `twenty_questions/` - 20 questions game
- `text_arena/` - Adversarial debate

**Purpose:** Train multiple agents that interact with each other.

**Core Library Features Used:**
- Multiple policies in same rollout
- Role-based conversation management
- Multi-agent reward assignment

**Recipe-Specific Code (Should Be Generalized):**
```python
class MultiAgentEnv:
    async def step(self, agent_id: str, action: Action) -> StepResult:
        """Execute action for specific agent."""

    def get_rewards(self) -> dict[str, float]:
        """Get final rewards for all agents."""
```

**Extractable Patterns:**
- Multi-agent environment protocol
- Turn-taking mechanics
- Joint/individual reward computation
- Agent role definitions

---

### 10. tool_use/search/ - Tool Use Training

**Location:** `/recipes/tool_use/search/`

**Files:**
- `train.py` - Training loop
- `search_env.py` - Search tool environment
- `tools.py` - Tool definitions and execution
- `embedding.py` - Document embedding for search
- `offline_eval.py` - Evaluation utilities

**Purpose:** Train models to use search tools effectively.

**Core Library Features Used:**
- Multi-turn environment with tool calls
- Tool call parsing from model output
- Observation modification after tool use

**Recipe-Specific Code (Should Be Generalized):**
```python
class ToolRegistry:
    def register(self, name: str, fn: Callable, schema: dict):
        """Register a tool with its schema."""

    def execute(self, name: str, args: dict) -> ToolResult:
        """Execute a registered tool."""
```

**Extractable Patterns:**
- Tool schema definitions
- Tool call parsing (model-specific)
- Tool execution sandboxing
- Observation augmentation

---

### 11. verifiers_rl/ - Verifier-Based RL

**Location:** `/recipes/verifiers_rl/`

**Files:**
- `train.py` - Training loop
- `verifiers_env.py` - Environment using external verifiers
- `tinker_openai.py` - OpenAI-compatible API wrapper
- `evaluate.py` - Evaluation utilities

**Purpose:** Use external verifiers for reward signals.

**Core Library Features Used:**
- Integration with `verifiers` library
- OpenAI-compatible API serving
- Multi-step verification

**Recipe-Specific Code (Should Be Generalized):**
```python
class VerifierEnv(Env):
    def __init__(self, verifier: Verifier):
        self.verifier = verifier

    async def step(self, action: Action) -> StepResult:
        result = await self.verifier.verify(action)
        return StepResult(reward=float(result.passed), ...)
```

**Extractable Patterns:**
- External verifier integration
- Step-by-step verification
- Verification result formatting

---

### 12. prompt_distillation/ - Prompt Distillation

**Location:** `/recipes/prompt_distillation/`

**Files:**
- `train.py` - Training to internalize prompts
- `create_data.py` - Generate training data

**Purpose:** Train models to follow instructions without explicit prompts.

**Core Library Features Used:**
- Different training weights for prompt vs response
- TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES

**Recipe-Specific Code:**
- Prompt removal from training data
- Instruction following verification

**Extractable Patterns:**
- Selective token weighting
- Prompt ablation studies

---

### 13. chat_sl/ - Chat Supervised Learning

**Location:** `/recipes/chat_sl/`

**Files:**
- `train.py` - Standard chat fine-tuning

**Purpose:** Fine-tune on multi-turn conversations.

**Core Library Features Used:**
- Full SL training pipeline
- Multi-turn conversation handling
- Various renderers (Llama3, Qwen3, etc.)

**Extractable Patterns:**
- Multi-turn data handling
- Turn-specific weighting

---

### 14. vlm_classifier/ - Vision-Language Classification

**Location:** `/recipes/vlm_classifier/`

**Files:**
- `train.py` - VLM fine-tuning
- `data.py` - Image dataset handling

**Purpose:** Train vision-language models for classification.

**Core Library Features Used:**
- Image processing pipeline
- `ImageChunk` in ModelInput
- Vision-specific renderers (Qwen3VL)

**Recipe-Specific Code (Should Be Generalized):**
```python
def image_to_datum(image_path: str, label: str, renderer: Renderer) -> Datum:
    """Convert image and label to training datum."""
```

**Extractable Patterns:**
- Image loading and preprocessing
- VLM-specific token handling
- Classification task formatting

---

### 15. rubric/ - Rubric-Based Evaluation

**Location:** `/recipes/rubric/`

**Files:**
- `train.py` - Training with rubric rewards
- `env.py` - Rubric evaluation environment
- `data.py` - Rubric dataset handling

**Purpose:** Train using structured rubric evaluations.

**Core Library Features Used:**
- LLM-as-judge for rubric scoring
- Multi-criteria evaluation
- Weighted rubric components

**Recipe-Specific Code (Should Be Generalized):**
```python
class RubricEvaluator:
    def __init__(self, rubric: dict[str, Criterion]):
        self.rubric = rubric

    async def evaluate(self, response: str) -> dict[str, float]:
        """Evaluate response against all criteria."""
```

**Extractable Patterns:**
- Rubric schema definition
- LLM-based scoring
- Multi-criteria aggregation

---

## Core Library Patterns

### 1. Renderer System

The renderer system handles model-specific chat templates:

```python
class Renderer(Protocol):
    def build_generation_prompt(self, messages: list[Message]) -> ModelInput
    def build_supervised_example(self, messages: list[Message]) -> (ModelInput, Tensor)
    def parse_response(self, tokens: list[int]) -> (Message, bool)
    def get_stop_sequences(self) -> list[str] | list[int]
```

**Implemented Renderers:**
- `RoleColonRenderer` - Simple "User: ... Assistant: ..." format
- `Llama3Renderer` - Meta Llama 3 format
- `Qwen3Renderer` - Qwen 3 with thinking support
- `Qwen3VLRenderer` - Qwen 3 Vision-Language
- `DeepSeekV3Renderer` - DeepSeek V3 format
- `GptOssRenderer` - OpenAI Harmony format
- `KimiK2Renderer` - Moonshot Kimi K2 format

**Extractable for Crucible:**
- Renderer behaviour definition
- Template registration system
- Token/weight generation

---

### 2. Environment System

The RL environment system:

```python
class Env:
    async def initial_observation(self) -> (Observation, StopCondition)
    async def step(self, action: Action) -> StepResult

class EnvGroupBuilder:
    async def make_envs(self) -> Sequence[Env]
    async def compute_group_rewards(self, trajectories, envs) -> list[(float, Metrics)]
```

**Key Types:**
- `Observation` = `ModelInput` - Prompt tokens
- `Action` = `list[int]` - Generated tokens
- `StopCondition` = `list[str] | list[int]` - Stop sequences
- `Trajectory` - Sequence of transitions
- `TrajectoryGroup` - Group of trajectories for training

**Extractable for Crucible:**
- Environment behaviour
- Trajectory types
- Group-based training abstraction

---

### 3. Training Loop Pattern

Both SL and RL use pipelined async training:

```python
# Pipelining pattern
pending_batch = None
for batch in dataset:
    submitted = await submit_batch(batch)  # Non-blocking
    if pending_batch:
        await finish_batch(pending_batch)   # Wait for previous
    pending_batch = submitted

if pending_batch:
    await finish_batch(pending_batch)
```

**Extractable for Crucible:**
- Pipelined execution abstraction
- Batch submission/completion pattern
- Progress tracking

---

### 4. Logging System

Multi-backend logging:

```python
class Logger(ABC):
    def log_hparams(self, config: Any) -> None
    def log_metrics(self, metrics: dict, step: int) -> None
    def log_long_text(self, key: str, text: str) -> None
    def close(self) -> None

class MultiplexLogger(Logger):
    """Forwards to multiple backends."""

# Backends: JsonLogger, WandbLogger, NeptuneLogger, TrackioLogger
```

**Extractable for Crucible:**
- Logger behaviour
- Multiplex pattern
- Backend adapters

---

### 5. Evaluator System

Pluggable evaluation during training:

```python
class TrainingClientEvaluator:
    async def __call__(self, training_client) -> dict[str, float]

class SamplingClientEvaluator:
    async def __call__(self, sampling_client) -> dict[str, float]

EvaluatorBuilder = Callable[[], Evaluator]
```

**Extractable for Crucible:**
- Evaluator behaviour
- Training vs sampling evaluator distinction
- Evaluation scheduling

---

## Extractable Functionality

### High Priority (Core Infrastructure)

1. **Training Loop Orchestration** (`crucible_train`)
   - Pipelined async execution
   - Checkpoint management
   - Progress tracking
   - Multi-epoch support

2. **Environment Abstractions** (`crucible_env`)
   - Env behaviour
   - EnvGroup behaviour
   - Trajectory types
   - Rollout execution

3. **Logging Infrastructure** (`crucible_log`)
   - Logger behaviour
   - Backend adapters (JSON, W&B, etc.)
   - Metrics aggregation
   - Logtree HTML reports

4. **Evaluation Framework** (`crucible_eval`)
   - Evaluator behaviour
   - Evaluation scheduling
   - Metric computation

### Medium Priority (Domain-Specific)

5. **Grading Utilities** (`crucible_grading`)
   - Answer extraction patterns
   - Numeric equivalence checking
   - Code extraction
   - Format validation

6. **Dataset Builders** (`crucible_data`)
   - HuggingFace integration (via snakebridge)
   - Batching and shuffling
   - Streaming support
   - Conversation formats

7. **Rubric Evaluation** (`crucible_rubric`)
   - Rubric schema
   - LLM-as-judge integration
   - Multi-criteria scoring

### Lower Priority (Can Use Snakebridge)

8. **Renderer System** - Keep in Python, access via snakebridge
9. **Model-Specific Templates** - Keep in Python
10. **Vision Processing** - Keep in Python

---

## Missing Abstractions

### 1. Unified Problem Interface

Currently `ProblemEnv` exists but could be more generic:

```elixir
defmodule Crucible.Problem do
  @callback get_question(t()) :: String.t()
  @callback check_answer(t(), String.t()) :: boolean()
  @callback check_format(t(), String.t()) :: boolean()
  @callback get_reference(t()) :: String.t()
end
```

### 2. Multi-Agent Coordination

No unified abstraction for multi-agent scenarios:

```elixir
defmodule Crucible.MultiAgent do
  @callback agents(t()) :: [agent_id()]
  @callback turn_order(t()) :: :sequential | :parallel | :custom
  @callback step(t(), agent_id(), action()) :: step_result()
  @callback compute_rewards(t()) :: %{agent_id() => float()}
end
```

### 3. Tool Registry

Tools are defined ad-hoc in each recipe:

```elixir
defmodule Crucible.Tool do
  @type t :: %{
    name: String.t(),
    description: String.t(),
    schema: map(),
    handler: (map() -> {:ok, term()} | {:error, term()})
  }

  @callback register(name, handler, schema) :: :ok
  @callback execute(name, args) :: result()
  @callback get_schema(name) :: schema()
end
```

### 4. Verifier Protocol

External verifiers need a standard interface:

```elixir
defmodule Crucible.Verifier do
  @callback verify(input :: term(), output :: term()) ::
    {:ok, %{passed: boolean(), reason: String.t()}} | {:error, term()}
end
```

### 5. Distillation Coordinator

Multi-teacher distillation needs orchestration:

```elixir
defmodule Crucible.Distillation do
  @callback teachers(t()) :: [teacher()]
  @callback select_teacher(t(), problem()) :: teacher()
  @callback generate(teacher(), prompt()) :: response()
  @callback aggregate(responses :: [response()]) :: response()
end
```

---

## Recommended Crucible Modules

Based on this analysis, the following Elixir modules should be created:

### 1. crucible_train
- Training loop orchestration
- Checkpoint management
- Progress tracking
- Async pipelining

### 2. crucible_env
- Environment behaviour
- EnvGroup behaviour
- Trajectory types
- Rollout execution
- Problem environment base

### 3. crucible_log
- Logger behaviour
- Backend adapters
- Logtree HTML reports
- Metrics aggregation

### 4. crucible_eval
- Evaluator behaviours
- Evaluation scheduling
- Metric computation
- NLL evaluation

### 5. crucible_grading
- Answer extraction
- Numeric equivalence
- Code extraction
- Format validation
- LLM-as-judge

### 6. crucible_tool
- Tool registry
- Schema definitions
- Execution sandboxing
- Result formatting

### 7. crucible_multiagent
- Multi-agent coordination
- Turn management
- Reward distribution
- Role definitions

### 8. crucible_preference
- Preference pair handling
- DPO/RLHF support
- Reference model management
- Reward margins

---

## Snakebridge Integration Points

The following should remain in Python and be accessed via snakebridge:

1. **Tinker SDK** - All API calls (`forward_backward`, `sample`, etc.)
2. **Renderers** - Model-specific chat templates
3. **Tokenizers** - HuggingFace tokenizers
4. **Image Processing** - PIL/vision processing
5. **Dataset Loading** - HuggingFace datasets
6. **NumPy/Torch Operations** - Tensor manipulation

Elixir should handle:

1. **Orchestration** - Training loop, checkpointing, scheduling
2. **State Management** - Experiment state, metrics history
3. **Configuration** - Experiment configs, hyperparameters
4. **Logging** - Structured logging, report generation
5. **Evaluation** - Metric computation, grading logic
6. **Multi-Agent** - Coordination, turn management

---

## Conclusion

The tinker-cookbook library provides rich functionality for ML training that can be extracted into the crucible ecosystem. The key insight is that Elixir should handle orchestration, state management, and logging while Python (via snakebridge) handles the heavy ML operations.

Priority should be given to:
1. Training loop abstraction (`crucible_train`)
2. Environment system (`crucible_env`)
3. Logging infrastructure (`crucible_log`)
4. Grading utilities (`crucible_grading`)

These provide the most value as pure Elixir modules and enable reliable, observable training pipelines.
