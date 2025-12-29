# Tinker-Cookbook Data Pipeline Analysis

**Date:** 2025-12-28
**Purpose:** Document data pipeline patterns in tinker-cookbook Python library and map to crucible_kitchen Elixir implementation.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Loading Patterns](#2-dataset-loading-patterns)
3. [Data Preprocessing and Transformation](#3-data-preprocessing-and-transformation)
4. [Tokenization Workflows](#4-tokenization-workflows)
5. [Batching and Collation](#5-batching-and-collation)
6. [Data Streaming and Caching](#6-data-streaming-and-caching)
7. [Format Conversions](#7-format-conversions)
8. [Elixir Implementation Mapping](#8-elixir-implementation-mapping)
9. [Gap Analysis](#9-gap-analysis)
10. [Recommendations](#10-recommendations)

---

## 1. Executive Summary

The tinker-cookbook library implements a sophisticated data pipeline for ML training with these key components:

| Component | Python Implementation | Elixir Status |
|-----------|----------------------|---------------|
| Dataset Loading | HuggingFace `datasets`, JSONL, blobfile | Partial (`DatasetBuilder`) |
| Chat Rendering | Multiple model-specific renderers | Complete (`Renderers`) |
| Tokenization | HuggingFace `transformers` | Adapter-specific (no port) |
| Batching | Custom `SupervisedDataset` abstraction | Partial |
| Streaming | `IterableDataset` support | Use DatasetStore + stream list |
| Format Conversion | JSONL native, Parquet via datasets | JSONL only |

**Critical for Training:** The data pipeline is the foundation for all training workflows (SFT, DPO, RL, Distillation). The pipeline must correctly:
1. Load raw data from various sources
2. Transform to chat message format
3. Render messages to tokens with model-specific templates
4. Create loss masks (weights) for training
5. Batch efficiently for GPU utilization

---

## 2. Dataset Loading Patterns

### 2.1 Python: HuggingFace Dataset Loading

**Location:** `tinker_cookbook/supervised/data.py`, `tinker_cookbook/recipes/chat_sl/chat_datasets.py`

```python
# Pattern 1: Load from HuggingFace Hub
class Tulu3Builder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)
        # ... return wrapped datasets
```

**Key Patterns:**
- Uses `datasets.load_dataset()` for HuggingFace Hub
- Supports split selection (train/test/validation)
- Shuffle with deterministic seed for reproducibility
- Take/skip for train/test splitting

### 2.2 Python: JSONL File Loading

**Location:** `tinker_cookbook/supervised/data.py`

```python
class FromConversationFileBuilder(ChatDatasetBuilder):
    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise ValueError(...)
                conversations.append(data)

        dataset = datasets.Dataset.from_list(conversations)
```

**Key Patterns:**
- Uses `blobfile` for cloud storage compatibility (GCS, S3, Azure)
- Validates required fields (`messages`)
- Converts to HuggingFace Dataset for consistency

### 2.3 Elixir: Current Implementation

**Location:** `lib/crucible_kitchen/dataset_builder.ex`

```elixir
# HuggingFace loading (requires hf_datasets_ex)
def from_hub(dataset_name, opts \\ []) do
  split = Keyword.get(opts, :split, "train")
  subset = Keyword.get(opts, :subset)

  case Code.ensure_loaded(HfDatasetsEx) do
    {:module, _} -> load_with_hf_datasets_ex(path, split, opts)
    {:error, _} -> {:error, :hf_datasets_ex_not_available}
  end
end

# JSONL loading
def from_jsonl(path) do
  case File.read(path) do
    {:ok, content} ->
      rows = content
        |> String.split("\n", trim: true)
        |> Enum.map(&Jason.decode!/1)
      {:ok, rows}
    {:error, reason} ->
      {:error, {:file_read_failed, reason}}
  end
end
```

**Gap:** No `blobfile` equivalent for cloud storage. Current implementation is local-only.

---

## 3. Data Preprocessing and Transformation

### 3.1 Python: Message Transformation Pipeline

**Location:** `tinker_cookbook/supervised/data.py`

```python
def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> tinker.Datum:
    """Common function to process a list of messages into a Datum."""
    tokens, weights = renderer.build_supervised_example(conversation, train_on_what=train_on_what)
    return datum_from_tokens_weights(tokens, weights, max_length)
```

**Flow:**
1. Take list of messages (chat format)
2. Render to tokens + weights using model-specific renderer
3. Truncate to max_length
4. Create Datum with input/target tokens and loss weights

### 3.2 Python: Datum Creation

**Location:** `tinker_cookbook/supervised/common.py`

```python
def datum_from_tokens_weights(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    max_length: int | None = None,
) -> tinker.Datum:
    if max_length is not None:
        tokens = tokens[:max_length]
    weights = weights[:max_length]

    input_tokens = tokens[:-1]  # All but last
    target_tokens = tokens[1:]   # All but first (shifted by 1)
    weights = weights[1:]        # Aligned with targets

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=input_tokens.tolist()),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=weights.tolist(), dtype="float32", shape=list(weights.shape)),
            "target_tokens": tinker.TensorData(data=[int(x) for x in target_tokens.tolist()], dtype="int64", shape=list(target_tokens.shape)),
        },
    )
```

**Critical:** Input/target shift by 1 for autoregressive training. Weights mask which tokens contribute to loss.

### 3.3 Python: TrainOnWhat Options

**Location:** `tinker_cookbook/renderers.py`

```python
class TrainOnWhat(StrEnum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    ALL_USER_AND_SYSTEM_MESSAGES = "all_user_and_system_messages"
```

**Usage:** Controls which tokens receive loss weight of 1.0 vs 0.0:
- `LAST_ASSISTANT_MESSAGE`: Only final assistant turn (common for instruction following)
- `ALL_ASSISTANT_MESSAGES`: All assistant turns (multi-turn training)
- `ALL_MESSAGES`: All turns including user (rare)
- `ALL_TOKENS`: Everything including headers (pretraining style)

### 3.4 Elixir: Current Implementation

**Location:** `lib/crucible_kitchen/renderers.ex`, `lib/crucible_kitchen/dataset_builder.ex`

```elixir
# TrainOnWhat in Elixir
defmodule CrucibleKitchen.Renderers.TrainOnWhat do
  def normalize(value) when is_atom(value), do: value
  def normalize("last_assistant_message"), do: :last_assistant_message
  # ... etc
end

# In Renderers.build_supervised_example/3
action_has_weight =
  case train_on_what do
    :last_assistant_message -> is_last and is_assistant
    :all_assistant_messages -> is_assistant
    :all_messages -> true
    :all_tokens -> true
    :all_user_and_system_messages -> is_user_or_system
    :customized -> message.trainable || false
  end
```

**Status:** Complete parity with Python implementation.

---

## 4. Tokenization Workflows

### 4.1 Python: Tokenizer Loading

**Location:** `tinker_cookbook/tokenizer_utils.py`

```python
@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # Avoid gating of Llama 3 models
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "baseten/Meta-Llama-3-tokenizer"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

**Key Patterns:**
- Cached loading for efficiency
- Workaround for gated models
- Fast tokenizer preferred

### 4.2 Python: Tokenization in Rendering

**Location:** `tinker_cookbook/renderers.py`

```python
class RoleColonRenderer(Renderer):
    def _render_message(self, message: Message) -> tuple[list[int], list[int], list[int]]:
        ob_str = message["role"].capitalize() + ":"
        ac_str = " " + message["content"] + "\n\n"
        ac_tail_str = "User:" if message["role"] == "assistant" else "<UNUSED>"
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_str, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )
```

**Pattern:** Each message rendered to 3 parts:
- `ob_part`: Observation/prefix (role header)
- `action_part`: Content tokens
- `action_tail`: Suffix for multi-turn (e.g., next role header)

### 4.3 Elixir: Current Implementation

**Location:** Adapter-specific helper (no port)

```elixir
defmodule CrucibleKitchen.Adapters.TokenizersEx do
  def load(_opts, model_or_path, _load_opts), do: Tokenizers.from_pretrained(model_or_path)
  def encode(_opts, tokenizer, text, opts), do: Tokenizers.encode(tokenizer, text, opts)
  def decode(_opts, tokenizer, ids, opts), do: Tokenizers.decode(tokenizer, ids, opts)
end
```

**Gap:** Helper needs concrete adapter usage in stages/renderers:
- `tokenizers-elixir` adapter (Rust-based, fast)
- Python bridge adapter (for exact HF parity)

---

## 5. Batching and Collation

### 5.1 Python: SupervisedDataset Abstraction

**Location:** `tinker_cookbook/supervised/types.py`

```python
class SupervisedDataset:
    def get_batch(self, index: int) -> list[tinker.Datum]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def set_epoch(self, seed: int = 0):
        """Shuffle differently each epoch."""
        logger.warning("set_epoch called, but shuffling is not implemented...")
```

### 5.2 Python: HuggingFace Dataset Wrapper

**Location:** `tinker_cookbook/supervised/data.py`

```python
class SupervisedDatasetFromHFDataset(SupervisedDataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        batch_size: int,
        map_fn: Callable[[dict], tinker.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset
        self.shuffle_dataset = hf_dataset
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn

    def get_batch(self, index: int) -> list[tinker.Datum]:
        rows = self.shuffle_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        if self.map_fn is not None:
            return [self.map_fn(row) for row in rows.to_list()]
        else:
            return [datum for row in rows.to_list() for datum in self.flatmap_fn(row)]

    def set_epoch(self, seed: int = 0):
        self.shuffle_dataset = self.hf_dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.hf_dataset) // self.batch_size
```

**Key Features:**
- `map_fn`: 1-to-1 transformation (one row -> one Datum)
- `flatmap_fn`: 1-to-many transformation (one row -> multiple Datum, used for DPO)
- Per-epoch shuffling with deterministic seeds
- Index-based batch retrieval

### 5.3 Elixir: Current Implementation

**Location:** `crucible_train/lib/crucible_train/ports/dataset_store.ex`

```elixir
@callback load_dataset(opts(), repo_id :: String.t(), keyword()) ::
            {:ok, dataset()} | {:error, term()}
@callback get_split(opts(), dataset(), String.t() | atom()) ::
            {:ok, dataset()} | {:error, term()}
@callback shuffle(opts(), dataset(), keyword()) ::
            {:ok, dataset()} | {:error, term()}
@callback take(opts(), dataset(), non_neg_integer()) ::
            {:ok, dataset()} | {:error, term()}
@callback skip(opts(), dataset(), non_neg_integer()) ::
            {:ok, dataset()} | {:error, term()}
@callback select(opts(), dataset(), Range.t() | [non_neg_integer()]) ::
            {:ok, dataset()} | {:error, term()}
@callback to_list(opts(), dataset()) ::
            {:ok, [map()]} | {:error, term()}
```

**Gap:** Use `CrucibleTrain.Supervised.Dataset` (already implemented) to provide:
- Batch indexing
- Per-epoch shuffling
- Map/flatmap transformation

---

## 6. Data Streaming and Caching

### 6.1 Python: Streaming Dataset Support

**Location:** `tinker_cookbook/supervised/data.py`

```python
class StreamingSupervisedDatasetFromHFDataset(SupervisedDataset):
    def __init__(
        self,
        hf_dataset: datasets.IterableDataset,
        batch_size: int,
        length: int,  # Must be provided for streaming
        map_fn: Callable[[dict], tinker.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,
        buffer_size: int = 10_000,
    ):
        self.hf_dataset = hf_dataset.shuffle(seed=0, buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_last_batch=True
        )
        self.dataset_iterator = iter(self.hf_dataset)
        self.index = -1
        self.length = length

    def get_batch(self, index: int) -> list[tinker.Datum]:
        assert index == self.index + 1  # Must be sequential
        self.index = index
        batch = next(self.dataset_iterator)
        rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        # ... apply map_fn or flatmap_fn

    def set_epoch(self, seed: int = 0):
        self.hf_dataset.set_epoch(seed)
        self.dataset_iterator = iter(self.hf_dataset)
        self.index = -1
```

**Key Features:**
- Shuffle buffer for streaming data
- Sequential-only access (no random indexing)
- Explicit length required
- Per-epoch iterator reset

### 6.2 Elixir: Current Implementation

```elixir
# In DatasetStore port
@callback stream(opts(), dataset(), stream_opts()) :: Enumerable.t()
```

**Gap:** Need streaming dataset wrapper with:
- Shuffle buffer implementation
- Sequential batch iteration
- Epoch management

---

## 7. Format Conversions

### 7.1 Python: JSONL Processing

**Location:** `tinker_cookbook/preference/preference_datasets.py`

```python
@chz.chz
class ComparisonBuilderFromJsonl(ComparisonDatasetBuilder):
    train_path: str
    test_path: str | None = None

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        import blobfile

        train_data = []
        with blobfile.BlobFile(self.train_path, "r", streaming=False) as f:
            for line in f:
                train_data.append(json.loads(line.strip()))

        train_dataset = datasets.Dataset.from_list(train_data)
        # ... similar for test
```

### 7.2 Python: Message Type Definition

**Location:** `tinker_cookbook/renderers.py`

```python
class ToolCall(TypedDict):
    name: str
    args: dict[str, str]

class Message(TypedDict):
    role: Role  # str: "user", "assistant", "system", "tool"
    content: str
    tool_calls: NotRequired[list[ToolCall]]
    thinking: NotRequired[str]  # For reasoning models
```

### 7.3 Python: Comparison/Preference Format

**Location:** `tinker_cookbook/preference/types.py`

```python
@dataclass
class Comparison:
    prompt_conversation: list[renderers.Message]
    completion_A: list[renderers.Message]
    completion_B: list[renderers.Message]

@dataclass
class LabeledComparison:
    comparison: Comparison
    label: Literal["A", "B", "Tie"]

    def swap(self) -> "LabeledComparison":
        return LabeledComparison(
            comparison=self.comparison.swap(),
            label={"A": "B", "B": "A", "Tie": "Tie"}[self.label],
        )
```

### 7.4 Elixir: Current Implementation

**Location:** `lib/crucible_kitchen/renderers/message.ex`

```elixir
defmodule CrucibleKitchen.Renderers.Message do
  defstruct [:role, :content, :tool_calls, :thinking, :trainable]

  def new(role, content, opts \\ []) do
    %__MODULE__{
      role: role,
      content: content,
      tool_calls: Keyword.get(opts, :tool_calls),
      thinking: Keyword.get(opts, :thinking),
      trainable: Keyword.get(opts, :trainable)
    }
  end

  def user(content), do: new("user", content)
  def assistant(content), do: new("assistant", content)
  def system(content), do: new("system", content)
  def tool(content, opts \\ []), do: new("tool", content, opts)
end
```

**Gap:** Need Comparison/LabeledComparison types for DPO workflows.

---

## 8. Elixir Implementation Mapping

### 8.1 Type Mapping

| Python Type | Elixir Type | Location |
|-------------|-------------|----------|
| `tinker.Datum` | `CrucibleKitchen.Types.Datum` | `lib/crucible_kitchen/types.ex` |
| `tinker.ModelInput` | `CrucibleKitchen.Types.ModelInput` | `lib/crucible_kitchen/types.ex` |
| `tinker.TensorData` | `CrucibleKitchen.Types.TensorData` | `lib/crucible_kitchen/types.ex` |
| `Message` | `CrucibleKitchen.Renderers.Message` | `lib/crucible_kitchen/renderers/message.ex` |
| `Renderer` | `CrucibleKitchen.Renderers` module | `lib/crucible_kitchen/renderers.ex` |
| `Comparison` | **MISSING** | - |
| `LabeledComparison` | **MISSING** | - |
| `SupervisedDataset` | **MISSING** (port only) | - |

### 8.2 Renderer Mapping

| Python Renderer | Elixir Renderer | Status |
|-----------------|-----------------|--------|
| `RoleColonRenderer` | `:role_colon` | Complete |
| `Llama3Renderer` | `:llama3` | Complete |
| `Qwen3Renderer` | `:qwen3` | Complete |
| `Qwen3DisableThinkingRenderer` | `:qwen3_no_thinking` | Complete |
| `Qwen3InstructRenderer` | `:qwen3_instruct` | **MISSING** |
| `DeepSeekV3Renderer` | **MISSING** | - |
| `DeepSeekV3DisableThinkingRenderer` | **MISSING** | - |
| `GptOssRenderer` | **MISSING** | - |
| `MistralRenderer` | `:mistral` | Complete |

### 8.3 Workflow Mapping

| Python Training | Elixir Workflow | Status |
|-----------------|-----------------|--------|
| `supervised/train.py` | `CrucibleKitchen.Workflows.Supervised` | Defined |
| `preference/train_dpo.py` | `CrucibleKitchen.Workflows.Preference` | Defined |
| `distillation/train_on_policy.py` | `CrucibleKitchen.Workflows.Distillation` | Defined |
| `rl/train.py` | `CrucibleKitchen.Workflows.Reinforcement` | Defined |

---

## 9. Gap Analysis

### 9.1 Critical Gaps (Block Training)

| Gap | Python Feature | Impact | Priority |
|-----|----------------|--------|----------|
| SupervisedDataset behaviour | Batch indexing, epoch shuffling | Cannot iterate training data correctly | P0 |
| Tokenizer adapter | HuggingFace tokenizer integration | Cannot tokenize text | P0 |
| HfDatasetsEx integration | Load from HuggingFace Hub | Limited to local files only | P1 |
| Cloud storage | blobfile (GCS, S3, Azure) | Cannot load from cloud | P1 |

### 9.2 Important Gaps (Limit Functionality)

| Gap | Python Feature | Impact | Priority |
|-----|----------------|--------|----------|
| Comparison types | DPO/preference data structures | Cannot run preference training | P1 |
| DeepSeek renderer | DeepSeekV3 chat template | Cannot train DeepSeek models | P2 |
| GPT-OSS renderer | OpenAI format with channels | Cannot train O1-style models | P2 |
| Streaming dataset | Large dataset support | Memory issues on large data | P2 |

### 9.3 Nice-to-Have Gaps

| Gap | Python Feature | Impact | Priority |
|-----|----------------|--------|----------|
| Parquet support | Arrow/Parquet loading | Must convert to JSONL | P3 |
| CSV enhanced | Pandas-like parsing | Limited CSV flexibility | P3 |
| Data validation | Schema enforcement | Manual validation needed | P3 |

---

## 10. Recommendations

### 10.1 Immediate Actions (P0)

1. **Use CrucibleTrain.Supervised.Dataset**

```elixir
defmodule CrucibleTrain.Supervised.Dataset do
  @callback get_batch(dataset :: term(), index :: non_neg_integer()) ::
              [CrucibleTrain.Types.Datum.t()]

  @callback length(dataset :: term()) :: non_neg_integer()

  @callback set_epoch(dataset :: term(), seed :: integer()) :: term()
end
```

2. **Implement tokenizer helper for tokenizers-elixir (adapter-specific)**

```elixir
defmodule CrucibleKitchen.Adapters.TokenizersEx do
  def load(_opts, model_or_path, _load_opts) do
    Tokenizers.from_pretrained(model_or_path)
  end

  def encode(_opts, tokenizer, text, opts) do
    add_special = Keyword.get(opts, :add_special_tokens, true)
    {:ok, encoding} = Tokenizers.encode(tokenizer, text)
    tokens = Tokenizers.Encoding.get_ids(encoding)
    # Handle add_special_tokens...
    {:ok, tokens}
  end
end
```

### 10.2 Short-Term Actions (P1)

1. **Add Comparison Types for DPO**

```elixir
defmodule CrucibleKitchen.Preference.Comparison do
  defstruct [:prompt_conversation, :completion_a, :completion_b]

  def swap(%__MODULE__{} = c) do
    %{c | completion_a: c.completion_b, completion_b: c.completion_a}
  end
end

defmodule CrucibleKitchen.Preference.LabeledComparison do
  defstruct [:comparison, :label]

  def swap(%__MODULE__{comparison: c, label: label}) do
    %__MODULE__{
      comparison: CrucibleKitchen.Preference.Comparison.swap(c),
      label: swap_label(label)
    }
  end

  defp swap_label("A"), do: "B"
  defp swap_label("B"), do: "A"
  defp swap_label("Tie"), do: "Tie"
end
```

2. **Integrate with hf_datasets_ex**

Ensure `hf_datasets_ex` is properly integrated for HuggingFace Hub dataset loading.

### 10.3 Medium-Term Actions (P2)

1. **Add DeepSeek and GPT-OSS Renderers**

Follow the pattern established for Llama3/Qwen3 renderers.

2. **Implement Streaming Dataset Wrapper**

Port the `StreamingSupervisedDatasetFromHFDataset` pattern to Elixir.

### 10.4 Architecture Notes

The tinker-cookbook data pipeline follows a clean layered architecture:

```
                  DATASET SOURCES
                 (HuggingFace, JSONL, CSV)
                         |
                         v
              DATASET BUILDERS (ChatDatasetBuilder)
              - Load raw data
              - Apply transforms
              - Split train/test
                         |
                         v
              SUPERVISED DATASET WRAPPER
              - Batch indexing
              - Epoch shuffling
              - map_fn / flatmap_fn
                         |
                         v
                    RENDERER
              - Message -> Tokens
              - Weights computation
                         |
                         v
                  DATUM CREATION
              - Input/target shift
              - TensorData packaging
                         |
                         v
                 TRAINING LOOP
              - Batch iteration
              - Forward/backward
              - Optimizer step
```

The Elixir implementation should maintain this separation of concerns, with:
- `DatasetBuilder` for loading/transforming
- `SupervisedDataset` behaviour for batch access
- `Renderers` for tokenization
- `Types.Datum` for training format

---

## Appendix A: Code References

### Python Files Analyzed

| File | Purpose |
|------|---------|
| `supervised/data.py` | Core dataset loading, HF wrapper, JSONL loading |
| `supervised/types.py` | SupervisedDataset, ChatDatasetBuilder abstractions |
| `supervised/common.py` | datum_from_tokens_weights, NLL computation |
| `supervised/train.py` | Full training loop with pipelining |
| `preference/types.py` | Comparison, LabeledComparison, PreferenceModel |
| `preference/preference_datasets.py` | ComparisonDatasetBuilder, JSONL loading |
| `preference/dpo_datasets.py` | DPO-specific dataset builder |
| `distillation/datasets.py` | PromptOnlyDataset, composite datasets |
| `recipes/chat_sl/chat_datasets.py` | Tulu3, NoRobots dataset builders |
| `recipes/preference/datasets.py` | HHH, UltraFeedback, Arena builders |
| `rl/types.py` | Env, Trajectory, RLDataset abstractions |
| `rl/data_processing.py` | Advantage computation, trajectory processing |
| `rl/problem_env.py` | ProblemEnv base class |
| `renderers.py` | All chat template renderers |
| `tokenizer_utils.py` | Tokenizer loading with caching |
| `utils/file_utils.py` | JSONL reading utility |

### Elixir Files Analyzed

| File | Purpose |
|------|---------|
| `lib/crucible_kitchen.ex` | Main module, workflow runner |
| `lib/crucible_kitchen/dataset_builder.ex` | Dataset loading utilities |
| `lib/crucible_kitchen/renderers.ex` | Chat template renderers |
| `lib/crucible_kitchen/types.ex` | Datum, ModelInput, TensorData |
| `crucible_train/lib/crucible_train/ports/dataset_store.ex` | Dataset port/behaviour |
| `lib/crucible_kitchen/adapters/noop/tokenizer_client.ex` | Tokenizer helper (no port) |

---

## Appendix B: Example Data Formats

### Chat Message Format (JSONL)

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "content": "2+2 equals 4."}
]}
```

### Preference Comparison Format (JSONL)

```json
{
  "comparison": {
    "prompt_conversation": [
      {"role": "user", "content": "Explain gravity."}
    ],
    "completion_A": [
      {"role": "assistant", "content": "Gravity is the force..."}
    ],
    "completion_B": [
      {"role": "assistant", "content": "It makes things fall."}
    ]
  },
  "label": "A"
}
```

### Tinker Datum Structure

```python
tinker.Datum(
    model_input=tinker.ModelInput.from_ints(tokens=[1, 2, 3, 4, 5]),
    loss_fn_inputs={
        "weights": tinker.TensorData(
            data=[0.0, 0.0, 1.0, 1.0, 1.0],
            dtype="float32",
            shape=[5]
        ),
        "target_tokens": tinker.TensorData(
            data=[2, 3, 4, 5, 6],  # Shifted by 1
            dtype="int64",
            shape=[5]
        )
    }
)
```

---

*Document generated by Claude Code analysis of tinker-cookbook and crucible_kitchen codebases.*
