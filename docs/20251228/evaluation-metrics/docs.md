# Tinker-Cookbook Evaluation and Metrics: Deep Dive

**Date:** 2025-12-28
**Purpose:** Comprehensive analysis of evaluation and metrics patterns in tinker-cookbook for Elixir crucible_ ecosystem integration.

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluator Architecture](#evaluator-architecture)
3. [Metric Computation Patterns](#metric-computation-patterns)
4. [Benchmark Integration (Inspect AI)](#benchmark-integration-inspect-ai)
5. [Evaluation Callbacks and Hooks](#evaluation-callbacks-and-hooks)
6. [Results Logging and Tracking](#results-logging-and-tracking)
7. [Comparison and Analysis Tools](#comparison-and-analysis-tools)
8. [Elixir Mapping and Gap Analysis](#elixir-mapping-and-gap-analysis)

---

## 1. Overview

The tinker-cookbook library provides a comprehensive evaluation framework designed for LLM training pipelines. The architecture separates concerns into:

- **Evaluators**: Abstract interfaces for different evaluation paradigms
- **Metrics**: Computation of specific measurements (KL divergence, NLL, accuracy)
- **Benchmarks**: Integration with external evaluation frameworks (Inspect AI)
- **Logging**: Multi-backend logging infrastructure for experiment tracking

### Core Philosophy

1. **Async-first design**: All evaluators are async to handle API latency
2. **Client-based architecture**: Evaluators receive either `TrainingClient` or `SamplingClient`
3. **Composable metrics**: Metrics can be computed independently and aggregated
4. **Environment-based RL**: Reinforcement learning uses explicit environment abstractions

---

## 2. Evaluator Architecture

### 2.1 Base Evaluator Classes

**File:** `/tinker_cookbook/eval/evaluators.py`

```python
class TrainingClientEvaluator:
    """
    An evaluator that takes in a TrainingClient.
    Used for evaluations requiring gradient access (e.g., NLL computation).
    """
    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        raise NotImplementedError


class SamplingClientEvaluator:
    """
    An evaluator that takes in a SamplingClient.
    Used for generation-based evaluations (e.g., accuracy, benchmarks).
    """
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        raise NotImplementedError


# Type aliases for flexibility
EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
```

### 2.2 Key Evaluator Types

| Evaluator | Base Class | Purpose |
|-----------|------------|---------|
| `NLLEvaluator` | `TrainingClientEvaluator` | Compute negative log-likelihood on held-out data |
| `CustomEvaluator` | `SamplingClientEvaluator` | Custom accuracy evaluation with grader function |
| `InspectEvaluator` | `SamplingClientEvaluator` | Run Inspect AI benchmark tasks |
| `RLTestSetEvaluator` | `SamplingClientEvaluator` | Evaluate RL policy on test environments |
| `ComparisonEvaluator` | `SamplingClientEvaluator` | Compare policy outputs using preference model |

### 2.3 NLL Evaluator Implementation

**File:** `/tinker_cookbook/supervised/nll_evaluator.py`

```python
class NLLEvaluator(TrainingClientEvaluator):
    def __init__(self, data: list[tinker.Datum]):
        self.data = data

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        return {"nll": nll}

    @classmethod
    def from_dataset(cls, dataset: SupervisedDataset) -> "NLLEvaluator":
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data)
```

### 2.4 Custom Evaluator Pattern

**File:** `/tinker_cookbook/eval/custom_evaluators.py`

```python
class CustomEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: Any,
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
    ):
        self.dataset = dataset
        self.grader_fn = grader_fn
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        metrics = {}
        num_examples = len(self.dataset)
        num_correct = 0

        sampling_params = types.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        for datum in self.dataset:
            model_input = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["input"])]
            )
            r = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            tokens = r.sequences[0].tokens
            response = self.renderer.parse_response(tokens)[0]
            if self.grader_fn(response["content"], datum["output"]):
                num_correct += 1

        metrics["accuracy"] = num_correct / num_examples
        return metrics
```

---

## 3. Metric Computation Patterns

### 3.1 KL Divergence Metrics

**File:** `/tinker_cookbook/rl/metrics.py`

The library computes multiple KL divergence variants:

```python
def compute_kl_sample_train(
    data_D: List[tinker.Datum], training_logprobs_D: List[torch.Tensor]
) -> Dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs."""
    all_diffs = []
    all_sampling_logprobs = []

    for datum, training_logprobs in safezip(data_D, training_logprobs_D):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0

        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)

    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()

    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
    }
```

### 3.2 KL Penalty Incorporation

```python
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place by:
    logp_base - logp_current - avg_kl
    """
    # Compute logprobs from base model
    full_sequence_inputs_D = [
        datum.model_input.append_int(datum.loss_fn_inputs["target_tokens"].data[-1])
        for datum in data_D
    ]
    base_logprobs_D = await asyncio.gather(*[
        base_sampling_client.compute_logprobs_async(seq)
        for seq in full_sequence_inputs_D
    ])

    # Compute differences and modify advantages
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    # ... advantage modification logic

    return {"kl_policy_base": float(avg_logp_diff)}
```

### 3.3 Post-Update KL Computation

```python
async def compute_post_kl(
    data_D: List[tinker.Datum], post_sampling_client: tinker.SamplingClient
) -> Dict[str, float]:
    """Compute post-update KL divergence metrics."""
    # Reconstruct original sequences
    full_sequence_inputs_D = [
        datum.model_input.append_int(datum.loss_fn_inputs["target_tokens"].data[-1])
        for datum in data_D
    ]

    new_logprobs_D = await asyncio.gather(*[
        post_sampling_client.compute_logprobs_async(seq)
        for seq in full_sequence_inputs_D
    ])

    # Compute KL between pre and post update
    prev_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    action_masks = [datum.loss_fn_inputs["mask"].to_torch() > 0 for datum in data_D]

    flat_diffs = [
        (prev_logprobs - torch.tensor(new_logprobs[1:]))[action_mask]
        for new_logprobs, prev_logprobs, action_mask in safezip(...)
    ]

    kl_post_v1 = torch.cat(flat_diffs).mean().item()
    kl_post_v2 = 0.5 * (torch.cat(flat_diffs)**2).mean().item()

    return {"kl_pre_post_v1": kl_post_v1, "kl_pre_post_v2": kl_post_v2}
```

### 3.4 NLL Computation

**File:** `/tinker_cookbook/supervised/common.py`

```python
def compute_mean_nll(
    logprobs_list: list[tinker.TensorData],
    weights_list: list[tinker.TensorData]
) -> float:
    """Compute weighted mean negative log likelihood."""
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    if total_weights == 0:
        return float("nan")

    return float(-total_weighted_logprobs / total_weights)
```

### 3.5 RL Trajectory Metrics

**File:** `/tinker_cookbook/rl/metric_util.py`

```python
def compute_trajectory_metrics(
    trajectory_groups_P: List[TrajectoryGroup], taglist_P: List[list[str]]
) -> Dict[str, float]:
    """Compute per-environment and aggregated metrics."""
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)

    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P)
        for trajgroups in tag2trajgroups.values()
    )

    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups).items()
            }
            out.update(prefixed_metrics)

    out.update({
        f"env/all/{k}": v
        for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()
    })
    return out


def _compute_trajectory_metrics(trajectory_groups_P: List[TrajectoryGroup]) -> Dict[str, float]:
    """Compute detailed metrics for trajectory groups."""
    flat_trajs = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]

    metrics = {
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs),
        "total_episodes": len(flat_trajs),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }

    metrics["reward/total"] = np.mean([
        reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()
    ]).item()

    return metrics
```

### 3.6 Math Answer Grading

**File:** `/tinker_cookbook/recipes/math_rl/math_grading.py`

Comprehensive math answer verification:

```python
def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    # First try mathd normalization
    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    # Then try more lenient normalization
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized == given_normalized:
        return True

    # Finally try sympy comparison
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    for gt_elem, given_elem in zip(ground_truth_elems, given_elems):
        is_correct = are_equal_under_sympy(gt_elem, given_elem)
        if not is_correct:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    """Use sympy to check mathematical equivalence."""
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            return simplified == 0
    except Exception:
        pass
    return False
```

---

## 4. Benchmark Integration (Inspect AI)

### 4.1 Inspect AI Model Wrapper

**File:** `/tinker_cookbook/eval/inspect_utils.py`

```python
@modelapi(name="tinker-sampling")
class InspectAPIFromTinkerSampling(InspectAIModelAPI):
    """Adapts tinker sampling clients to the inspect API interface."""

    def __init__(
        self,
        renderer_name: str,
        model_name: str,
        model_path: str | None = None,
        sampling_client: tinker.SamplingClient | None = None,
        config: InspectAIGenerateConfig = InspectAIGenerateConfig(),
        verbose: bool = False,
    ):
        super().__init__(model_name=model_name, config=config)

        if sampling_client is not None:
            self.sampling_client = sampling_client
        elif model_path is not None:
            service_client = tinker.ServiceClient()
            self.sampling_client = service_client.create_sampling_client(model_path=model_path)

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def generate(
        self,
        input: list[InspectAIChatMessage],
        tools: list[InspectAIToolInfo],
        tool_choice: InspectAIToolChoice,
        config: InspectAIGenerateConfig,
    ) -> InspectAIModelOutput:
        """Main interface for Inspect AI model evaluation."""
        if config.system_message:
            input = [ChatMessageSystem(content=config.system_message)] + input

        convo = convert_inspect_messages(input)
        prompt = self.renderer.build_generation_prompt(convo)

        sampling_params = tinker.SamplingParams(
            temperature=config.temperature or 1.0,
            max_tokens=config.max_tokens or 128,
            stop=self.renderer.get_stop_sequences(),
        )

        sample_result = await self.sampling_client.sample_async(
            prompt=prompt, sampling_params=sampling_params, num_samples=1
        )

        # Parse and format response
        parsed_responses = [
            self.renderer.parse_response(r.tokens)[0]
            for r in sample_result.sequences
        ]

        return InspectAIModelOutput(
            model=self.model_name,
            choices=[...],
            usage=get_model_usage(prompt.to_ints(), sample_result.sequences)
        )
```

### 4.2 Inspect Evaluator Configuration

**File:** `/tinker_cookbook/eval/inspect_evaluators.py`

```python
@chz.chz
class InspectEvaluatorBuilder:
    """Configuration for inspect evaluation."""

    # Required parameters
    tasks: Tasks
    renderer_name: str
    model_name: str | None = None

    # Generation parameters
    temperature: float = 1.0
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: int = -1

    # Evaluation parameters
    limit: Optional[int] = None
    debug_errors: bool = True
    log_dir: Optional[str] = None
    max_connections: int = 512
    log_level: str = "INFO"

    def __call__(self) -> SamplingClientEvaluator:
        return InspectEvaluator(self)


class InspectEvaluator(SamplingClientEvaluator):
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        # Create inspect API wrapper
        api = InspectAPIFromTinkerSampling(
            renderer_name=self.config.renderer_name,
            model_name=self.config.model_name,
            sampling_client=sampling_client,
        )

        model = InspectAIModel(api=api, config=InspectAIGenerateConfig(...))

        # Run evaluation
        results = await eval_async(
            tasks=self.config.tasks,
            model=[model],
            limit=self.config.limit,
            retry_on_error=0,
            fail_on_error=False,
        )

        # Extract metrics
        metrics = {}
        for task_result in results:
            if task_result.results is not None:
                for task_name, score in task_result.results.scores[0].metrics.items():
                    dataset_name = task_result.eval.dataset.name or "unknown"
                    metrics[f"{dataset_name}/{task_name}"] = score.value

        return metrics
```

### 4.3 LLM-as-Judge Pattern

**File:** `/tinker_cookbook/eval/custom_inspect_task.py`

```python
@task
def example_lm_as_judge() -> Task:
    """Example task using LLM-as-a-judge scoring."""
    return Task(
        name="llm_as_judge",
        dataset=QA_DATASET,
        solver=generate(),
        scorer=model_graded_qa(
            instructions="Grade strictly against the target text...",
            partial_credit=False,
            model=GRADER_MODEL,
        ),
    )
```

---

## 5. Evaluation Callbacks and Hooks

### 5.1 Training Loop Integration

**File:** `/tinker_cookbook/rl/train.py`

```python
@scope
async def run_single_evaluation(evaluator, cfg, i_batch, sampling_client):
    """Run a single evaluator with logging scope."""
    ev_name = _get_evaluator_name(evaluator)
    with _get_logtree_scope(
        log_path=cfg.log_path,
        num_groups_to_log=cfg.num_groups_to_log,
        f_name=f"eval_{ev_name}_iteration_{i_batch:06d}",
        scope_name=f"Running evaluation {ev_name} {i_batch}",
    ):
        eval_metrics = await evaluator(sampling_client)
        return {f"test/{k}": v for k, v in eval_metrics.items()}


@scope
async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: tinker.SamplingClient,
    cfg: Config,
    i_batch: int,
) -> dict[str, Any]:
    """Run all evaluators in parallel and return aggregated metrics."""
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        task = asyncio.create_task(
            run_single_evaluation(evaluator, cfg, i_batch, sampling_client),
            name=f"eval_{ev_name or i}_iteration_{i_batch:06d}",
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    metrics = {}
    for result in results:
        metrics.update(result)
    return metrics
```

### 5.2 Evaluation Scheduling

In the training loop:

```python
# In do_sync_training()
for i_batch in range(start_batch, end_batch):
    # Run evaluations at configured intervals
    if (cfg.eval_every > 0 and i_batch % cfg.eval_every == 0) or i_batch == end_batch - 1:
        with timed("run_evals", metrics):
            eval_metrics = await run_evaluations_parallel(
                evaluators, sampling_client, cfg, i_batch
            )
            metrics.update(eval_metrics)
```

### 5.3 Supervised Training Evaluation Hooks

**File:** `/tinker_cookbook/supervised/train.py`

```python
async def run_evals(
    evaluators: list[Evaluator],
    training_client: tinker.TrainingClient,
    step: int,
) -> dict[str, float]:
    """Run all evaluators and return metrics with test/ prefix."""
    metrics = {}
    sampling_client = None

    for evaluator in evaluators:
        if isinstance(evaluator, TrainingClientEvaluator):
            eval_metrics = await evaluator(training_client)
        elif isinstance(evaluator, SamplingClientEvaluator):
            # Lazy sampling client creation
            if sampling_client is None:
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                    f"evals_step_{step}"
                )
            eval_metrics = await evaluator(sampling_client)

        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

    return metrics
```

### 5.4 Infrequent Evaluation Pattern

```python
# In supervised training config
@chz.chz
class Config:
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    eval_every: int = 10
    infrequent_eval_every: int = 100

# In training loop
if evaluators and config.eval_every > 0 and step % config.eval_every == 0:
    eval_metrics = await run_evals(evaluators, training_client, step)

if infrequent_evaluators and config.infrequent_eval_every > 0 and step % config.infrequent_eval_every == 0:
    eval_metrics = await run_evals(infrequent_evaluators, training_client, step)
```

---

## 6. Results Logging and Tracking

### 6.1 Logger Architecture

**File:** `/tinker_cookbook/utils/ml_log.py`

```python
class Logger(ABC):
    """Abstract base class for loggers."""

    @abstractmethod
    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters/configuration."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics dictionary with optional step number."""
        pass

    def log_long_text(self, key: str, text: str) -> None:
        """Log long text content (optional)."""
        pass

    def close(self) -> None:
        """Cleanup when done."""
        pass

    def sync(self) -> None:
        """Force synchronization."""
        pass

    def get_logger_url(self) -> str | None:
        """Get a permalink to view results."""
        return None
```

### 6.2 Available Logger Implementations

| Logger | Backend | Features |
|--------|---------|----------|
| `JsonLogger` | JSONL files | Local storage, config.json, metrics.jsonl |
| `PrettyPrintLogger` | Console | Rich table formatting |
| `WandbLogger` | Weights & Biases | Cloud tracking, visualization |
| `NeptuneLogger` | Neptune | Cloud experiment tracking |
| `TrackioLogger` | Trackio | Alternative cloud tracking |
| `MultiplexLogger` | Multiple | Combines multiple backends |

### 6.3 JSON Logger Implementation

```python
class JsonLogger(Logger):
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_hparams(self, config: Any) -> None:
        if not self._logged_hparams:
            config_dict = dump_config(config)
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            # Also save code diff
            with open(self.log_dir / "code.diff", "w") as f:
                f.write(code_state())
            self._logged_hparams = True

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        log_entry = {"step": step} if step is not None else {}
        log_entry.update(metrics)
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
```

### 6.4 Multiplex Logger

```python
class MultiplexLogger(Logger):
    """Logger that forwards operations to multiple child loggers."""

    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def get_logger_url(self) -> str | None:
        for logger in self.loggers:
            if url := logger.get_logger_url():
                return url
        return None
```

### 6.5 Setup Function

```python
def setup_logging(
    log_dir: str,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    config: Any | None = None,
) -> Logger:
    """Set up logging infrastructure with multiple backends."""
    loggers = []

    # Always add JSON logger
    loggers.append(JsonLogger(log_dir))

    # Always add pretty print
    loggers.append(PrettyPrintLogger())

    # Add cloud loggers if configured
    if wandb_project and _wandb_available:
        loggers.append(WandbLogger(project=wandb_project, config=config))

    if wandb_project and _neptune_available:
        loggers.append(NeptuneLogger(project=wandb_project, config=config))

    ml_logger = MultiplexLogger(loggers)

    if config is not None:
        ml_logger.log_hparams(config)

    return ml_logger
```

---

## 7. Comparison and Analysis Tools

### 7.1 Preference Comparison Evaluator

**File:** `/tinker_cookbook/preference/comparison_policy_evaluator.py`

```python
class ComparisonEvaluator(SamplingClientEvaluator):
    """Evaluates a policy by comparing completions to references using a reward model."""

    def __init__(
        self,
        preference_model_builder: Callable[[], PreferenceModel],
        comparisons: Sequence[Comparison],
        renderer_name: str,
        model_name_for_tokenizer: str,
        both_ways: bool = True,
        max_tokens: int = 1024,
    ):
        self.preference_model_builder = preference_model_builder
        self.comparisons = comparisons
        self.renderer = get_renderer(renderer_name, get_tokenizer(model_name_for_tokenizer))
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        preference_model = self.preference_model_builder()
        policy = TinkerMessageCompleter(sampling_client, self.renderer, self.max_tokens)

        async def process_comparison(comparison: Comparison) -> float:
            new_completion_message = await policy(comparison.prompt_conversation)
            new_comparison = replace(comparison, completion_B=[new_completion_message])

            # Get scores in both directions
            r_0, r_1 = await asyncio.gather(
                preference_model(new_comparison),
                preference_model(new_comparison.swap())
            )
            # Normalize to 0-1 range
            return (r_0 - r_1 + 2) / 4.0

        results = await asyncio.gather(*[
            process_comparison(c) for c in self.comparisons
        ])

        return {
            "win_rate": np.mean(results).item(),
            "stderr": np.std(results).item() / np.sqrt(len(results)),
        }
```

### 7.2 RL Test Set Evaluation

**File:** `/tinker_cookbook/rl/metric_util.py`

```python
class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: RLDataset,
        max_tokens: int,
        name: str | None = None,
        num_groups_to_log: int = 4,
    ):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens
        self.name = name
        self.num_groups_to_log = num_groups_to_log

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)

        async def run_group_rollout(builder, i):
            enable_logging = i < self.num_groups_to_log
            with logtree.optional_enable_logging(enable=enable_logging):
                return await do_group_rollout(builder, policy)

        trajectory_groups_P = await asyncio.gather(*[
            run_group_rollout(builder, i)
            for i, builder in enumerate(self.env_group_builders_P)
        ])

        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)

        if self.name is not None:
            metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
```

### 7.3 Offline Evaluation (Tool Use)

**File:** `/tinker_cookbook/recipes/tool_use/search/offline_eval.py`

```python
async def evaluate_one_dataset(data: list[SearchR1Datum], args) -> dict:
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    sampling_client = service_client.create_sampling_client(
        model_path=args.tinker_checkpoint_url
    )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=args.max_tokens)

    async def evaluate_single_item(item) -> EvaluationResult:
        env = SearchEnv(item["question"], item["answer"], ...)
        trajectory = await do_single_rollout(policy, env)
        correct_score = trajectory.transitions[-1].metrics.get("correct", 0.0)
        return {"question": item["question"], "correct_score": correct_score}

    results = await asyncio.gather(*[evaluate_single_item(item) for item in data])

    correct_scores = [r["correct_score"] for r in results]
    return {
        "total_samples": len(correct_scores),
        "total_correct": sum(correct_scores),
        "accuracy": sum(correct_scores) / len(correct_scores),
    }
```

### 7.4 Verifiers RL Evaluation

**File:** `/tinker_cookbook/recipes/verifiers_rl/evaluate.py`

```python
def evaluate(
    vf_env_id: str,
    vf_env_args: dict,
    model_name: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
):
    env = vf.load_environment(vf_env_id, **vf_env_args)

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling = service.create_sampling_client(base_model=model_name)
    client = TinkerAsyncOpenAIClient(sampling, renderer, tokenizer)

    results = env.evaluate_sync(
        client=client,
        model=model_name,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args={"max_tokens": max_tokens, "temperature": temperature},
    )

    log_results(results, ...)


def log_results(results, ...):
    print(f"reward: avg - {sum(results.reward) / len(results.reward):.3f}")

    for k in results.metrics:
        v = results.metrics[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
```

---

## 8. Elixir Mapping and Gap Analysis

### 8.1 Current Elixir Implementation

**crucible_datasets** (`/crucible_datasets/lib/dataset_manager/`):

| Component | Status | Implementation |
|-----------|--------|----------------|
| `EvaluationResult` | Implemented | Struct with accuracy, metrics, item_results |
| `Evaluator.ExactMatch` | Implemented | String/numeric/list comparison |
| Dataset Loaders | Partial | Basic structure exists |

**crucible_kitchen** (`/crucible_kitchen/lib/crucible_kitchen/`):

| Component | Status | Implementation |
|-----------|--------|----------------|
| Stage behaviour | Implemented | `validate/1`, `execute/1`, `rollback/2` |
| `Stages.Evaluate` | Stub | Empty implementation |
| Metric recording | Implemented | `record_metric/3,4` in Stage helpers |
| Workflows | Partial | Preference, Reinforcement, Distillation stubs |

### 8.2 Mapping Table: Python to Elixir

| Python (tinker-cookbook) | Elixir (crucible_) | Gap Status |
|--------------------------|-------------------|------------|
| `TrainingClientEvaluator` | `CrucibleKitchen.Evaluator.Training` | **Missing** |
| `SamplingClientEvaluator` | `CrucibleKitchen.Evaluator.Sampling` | **Missing** |
| `NLLEvaluator` | `CrucibleKitchen.Evaluator.NLL` | **Missing** |
| `CustomEvaluator` | `CrucibleKitchen.Evaluator.Custom` | **Missing** |
| `InspectEvaluator` | N/A (external) | **Not applicable** |
| `RLTestSetEvaluator` | `CrucibleKitchen.Evaluator.RLTestSet` | **Missing** |
| `ComparisonEvaluator` | `CrucibleKitchen.Evaluator.Comparison` | **Missing** |
| `compute_kl_sample_train` | `CrucibleKitchen.Metrics.KL` | **Missing** |
| `compute_mean_nll` | `CrucibleKitchen.Metrics.NLL` | **Missing** |
| `compute_trajectory_metrics` | `CrucibleKitchen.Metrics.Trajectory` | **Missing** |
| `grade_answer` | `CrucibleDatasets.Evaluator.MathGrading` | **Missing** |
| `Logger` (abstract) | `CrucibleKitchen.Logger` | **Missing** |
| `JsonLogger` | `CrucibleKitchen.Logger.JSON` | **Missing** |
| `WandbLogger` | `CrucibleKitchen.Logger.Wandb` | **Missing** |
| `MultiplexLogger` | `CrucibleKitchen.Logger.Multiplex` | **Missing** |
| `setup_logging` | `CrucibleKitchen.Logger.setup/1` | **Missing** |
| `ExactMatch.compute/2` | `CrucibleDatasets.Evaluator.ExactMatch` | **Implemented** |
| `EvaluationResult` | `CrucibleDatasets.EvaluationResult` | **Implemented** |

### 8.3 Priority Implementation Recommendations

#### Phase 1: Core Evaluator Framework (High Priority)

1. **Evaluator Behaviours**
```elixir
defmodule CrucibleKitchen.Evaluator do
  @type result :: {:ok, map()} | {:error, term()}

  @callback evaluate(client :: term(), opts :: keyword()) :: result()
end

defmodule CrucibleKitchen.Evaluator.Training do
  @behaviour CrucibleKitchen.Evaluator
end

defmodule CrucibleKitchen.Evaluator.Sampling do
  @behaviour CrucibleKitchen.Evaluator
end
```

2. **NLL Evaluator**
```elixir
defmodule CrucibleKitchen.Evaluator.NLL do
  @moduledoc "Compute negative log-likelihood on held-out data."
  use CrucibleKitchen.Evaluator.Training

  def evaluate(training_client, opts) do
    data = Keyword.fetch!(opts, :data)
    # Compute weighted mean NLL using Nx
    {:ok, %{nll: nll_value}}
  end
end
```

#### Phase 2: Metrics Library (High Priority)

3. **KL Divergence Metrics**
```elixir
defmodule CrucibleKitchen.Metrics.KL do
  @moduledoc "KL divergence computation utilities."

  def kl_sample_train(data, training_logprobs) do
    # Compute KL v1, v2, and entropy
    %{
      kl_sample_train_v1: kl_v1,
      kl_sample_train_v2: kl_v2,
      entropy: entropy
    }
  end

  def kl_post_update(data, sampling_client) do
    # Compute post-update KL
    %{kl_pre_post_v1: kl_v1, kl_pre_post_v2: kl_v2}
  end
end
```

4. **Trajectory Metrics**
```elixir
defmodule CrucibleKitchen.Metrics.Trajectory do
  @moduledoc "RL trajectory metrics computation."

  def compute(trajectory_groups, tag_lists) do
    # Aggregate by tags and compute:
    # - ac_tokens_per_turn
    # - ob_tokens_per_turn
    # - turns_per_episode
    # - reward/total
    %{}
  end
end
```

#### Phase 3: Logging Infrastructure (Medium Priority)

5. **Logger Behaviour**
```elixir
defmodule CrucibleKitchen.Logger do
  @callback log_hparams(config :: map()) :: :ok
  @callback log_metrics(metrics :: map(), step :: integer() | nil) :: :ok
  @callback close() :: :ok
  @optional_callbacks [close: 0]
end
```

6. **JSON Logger**
```elixir
defmodule CrucibleKitchen.Logger.JSON do
  @behaviour CrucibleKitchen.Logger
  use GenServer

  def log_metrics(metrics, step) do
    entry = Map.put(metrics, :step, step)
    File.write!(@metrics_file, Jason.encode!(entry) <> "\n", [:append])
  end
end
```

7. **Multiplex Logger**
```elixir
defmodule CrucibleKitchen.Logger.Multiplex do
  @behaviour CrucibleKitchen.Logger

  def log_metrics(metrics, step) do
    Enum.each(@loggers, &(&1.log_metrics(metrics, step)))
  end
end
```

#### Phase 4: Advanced Evaluators (Lower Priority)

8. **Comparison Evaluator** (for DPO/RLHF)
9. **RL Test Set Evaluator**
10. **Math Answer Grading** (extend ExactMatch)

### 8.4 Architecture Recommendations

1. **Async-first Design**: Use `Task.async_stream/3` for parallel evaluation
2. **Client Abstraction**: Define `TrainingClient` and `SamplingClient` behaviours in Tinkex
3. **Telemetry Integration**: Emit `:telemetry` events for all evaluations
4. **PubSub for Streaming**: Use Phoenix.PubSub for real-time metric updates
5. **Nx for Numerics**: Use Nx tensors for KL/NLL computation (matches Python torch)

### 8.5 Example: Full Evaluation Stage

```elixir
defmodule CrucibleKitchen.Stages.Evaluate do
  @moduledoc "Run configured evaluators."
  use CrucibleKitchen.Stage

  def name, do: :evaluate

  def execute(context) do
    evaluators = get_config(context, :evaluators, [])
    sampling_client = get_state(context, :sampling_client)
    step = get_state(context, :global_step, 0)

    # Run evaluators in parallel
    results =
      evaluators
      |> Task.async_stream(&(&1.evaluate(sampling_client, step: step)))
      |> Enum.reduce(%{}, fn {:ok, metrics}, acc -> Map.merge(acc, metrics) end)

    # Prefix with test/
    metrics = for {k, v} <- results, into: %{}, do: {"test/#{k}", v}

    # Record and emit telemetry
    context =
      Enum.reduce(metrics, context, fn {k, v}, ctx ->
        record_metric(ctx, k, v, step: step)
      end)

    :telemetry.execute(
      [:crucible_kitchen, :evaluation, :complete],
      metrics,
      %{step: step}
    )

    {:ok, context}
  end
end
```

---

## Summary

The tinker-cookbook library provides a mature, well-designed evaluation framework with:

1. **Clean abstractions**: `TrainingClientEvaluator` vs `SamplingClientEvaluator`
2. **Comprehensive metrics**: KL divergence, NLL, trajectory metrics, math grading
3. **Flexible logging**: Multi-backend with JSONL, W&B, Neptune support
4. **Benchmark integration**: Inspect AI for standardized evaluations
5. **Async-first**: All evaluators are async for API efficiency

The Elixir ecosystem has foundational pieces (`EvaluationResult`, `ExactMatch`) but significant gaps in:

- Evaluator behaviours and implementations
- KL/NLL metric computation
- Logging infrastructure
- Training loop integration

Priority should be given to:
1. Evaluator behaviours (matches Python pattern)
2. NLL evaluator (most commonly used)
3. KL metrics (critical for RL training)
4. JSON/Multiplex loggers (experiment tracking)

This mapping enables incremental migration while maintaining compatibility with the Python tinker ecosystem.
