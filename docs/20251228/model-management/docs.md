# Model Management in tinker-cookbook: Comprehensive Analysis

## Executive Summary

This document provides a deep analysis of model management patterns in the Python tinker-cookbook library, mapping each pattern to the Elixir crucible_ ecosystem (tinkex, hf_hub_ex, crucible_kitchen) and identifying implementation gaps.

**Key Insight**: Tinker is a cloud-based training service where models live server-side. The "cookbook" handles model references, tokenization, rendering, and checkpoint management, but NOT local model loading or quantization. This fundamentally differs from traditional local model management.

---

## Table of Contents

1. [Model Loading Patterns](#1-model-loading-patterns)
2. [Model Configuration Management](#2-model-configuration-management)
3. [Quantization Patterns](#3-quantization-patterns)
4. [Adapter Management (LoRA)](#4-adapter-management-lora)
5. [Model Conversion and Export](#5-model-conversion-and-export)
6. [Model Registry Patterns](#6-model-registry-patterns)
7. [Tokenizer Management](#7-tokenizer-management)
8. [Gap Analysis Summary](#8-gap-analysis-summary)
9. [Recommended Elixir Implementation Roadmap](#9-recommended-elixir-implementation-roadmap)

---

## 1. Model Loading Patterns

### 1.1 How tinker-cookbook Loads Models

In tinker-cookbook, models are NOT loaded locally. Instead, models are referenced by their HuggingFace identifiers and the Tinker service handles GPU-side model loading.

#### Python Code (from `training-sampling.mdx`):

```python
import tinker

service_client = tinker.ServiceClient()

# List available models
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)

# Create a training client with a base model reference
base_model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
```

#### Python Code (from `model_info.py`):

```python
# File: tinker_cookbook/model_info.py
"""
Simple static data about models.

We centralize it here so that the cookbook doesn't reference model names (except in notebooks or tests).
"""

# Model identifiers are just HuggingFace repo names
LLAMA3_1_8B = "meta-llama/Llama-3.1-8B"
LLAMA3_1_70B = "meta-llama/Llama-3.1-70B"
QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B"
# ... more constants

# Architecture metadata
MODEL_INFO = {
    "meta-llama/Llama-3.1-8B": {
        "architecture": "llama",
        "hidden_size": 4096,
        "num_layers": 32,
    },
    # ...
}
```

### 1.2 Elixir Implementation (Tinkex)

The Elixir `Tinkex` library mirrors this pattern:

```elixir
# File: tinkex/lib/tinkex/service_client.ex
defmodule Tinkex.ServiceClient do
  def create_lora_training_client(service_client, opts \\ []) do
    GenServer.call(service_client, {:create_training_client, opts})
  end
end

# File: tinkex/lib/tinkex/training_client.ex
defmodule Tinkex.TrainingClient do
  defp ensure_model(opts, session_id, model_seq_id, config, service_api) do
    case opts[:model_id] do
      model_id when is_binary(model_id) ->
        {:ok, model_id}
      _ ->
        with {:ok, base_model} <- fetch_base_model(opts),
             {:ok, response} <-
               service_api.create_model(
                 %CreateModelRequest{
                   session_id: session_id,
                   model_seq_id: model_seq_id,
                   base_model: base_model,
                   lora_config: Keyword.get(opts, :lora_config, %LoraConfig{})
                 },
                 config: config
               ) do
          {:ok, parse_model_id(response)}
        end
    end
  end
end
```

### 1.3 HuggingFace Hub Integration (hf_hub_ex)

For downloading model files locally (tokenizers, configs):

```elixir
# File: hf_hub_ex/lib/hf_hub/download.ex
defmodule HfHub.Download do
  @spec hf_hub_download(download_opts()) :: {:ok, Path.t()} | {:error, term()}
  def hf_hub_download(opts) do
    repo_id = Keyword.fetch!(opts, :repo_id)
    filename = Keyword.fetch!(opts, :filename)
    repo_type = Keyword.get(opts, :repo_type, :model)
    revision = Keyword.get(opts, :revision, "main")

    cache_path = HfHub.FS.file_path(repo_id, repo_type, filename, revision)

    if File.exists?(cache_path) and not force_download do
      {:ok, cache_path}
    else
      url = build_download_url(repo_id, repo_type, filename, revision)
      do_download_file(url, cache_path, token)
    end
  end
end
```

### 1.4 Gaps in Elixir Implementation

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| Model reference by HF ID | Yes | Yes (Tinkex) | None |
| List available models | `get_server_capabilities()` | Not implemented | **GAP** |
| Model info constants | `model_info.py` | None | **GAP** |
| Snapshot download | `hf_hub_download` | `HfHub.Download` | Partial |
| Model config parsing | `transformers.AutoConfig` | None | **GAP** |

---

## 2. Model Configuration Management

### 2.1 How tinker-cookbook Manages Configs

Configuration happens at two levels:

1. **Training Configuration** - Learning rate, batch size, etc.
2. **Model Configuration** - LoRA rank, architecture settings

#### Python Code (from `hyperparam_utils.py`):

```python
# File: tinker_cookbook/hyperparam_utils.py
"""
This module contains utilities for hyperparameter tuning.
"""

def get_lr(model_name: str, batch_size: int = 1) -> float:
    """
    Get the learning rate for a model.

    The learning rate is computed using the formula:
        lr = base_lr * sqrt(batch_size)

    where base_lr is a model-specific constant (typically around 1e-5 for LoRA).
    """
    base_lr = _get_base_lr(model_name)
    return base_lr * math.sqrt(batch_size)

def get_lora_lr_over_full_finetune_lr(model_name: str) -> float:
    """
    Get the factor by which LoRA LR should be multiplied vs full fine-tuning LR.

    This depends on the model's hidden size - larger models need larger multipliers.
    LoRA typically needs 20-100x higher LR than full fine-tuning.
    """
    # Llama-3.2-1B: factor is 32
    # Llama-3.1-70B: factor is 128
    hidden_size = _get_hidden_size(model_name)
    return 4 * math.sqrt(hidden_size)

def get_lora_param_count(model_name: str, lora_rank: int = 32) -> int:
    """
    Get the number of trainable parameters for LoRA at a given rank.
    """
    # LoRA adds low-rank matrices A (r x d) and B (d x r) to each weight
    pass
```

#### Python LoRA Config (from API types):

```python
# From tinker/types
class LoraConfig(StrictBase):
    """LoRA configuration for model fine-tuning."""
    rank: int = 32  # LoRA rank (dimension of low-rank matrices)
    seed: int | None = None  # Seed for reproducible initialization
    train_unembed: bool = True  # Add LoRA to unembedding layer
    train_mlp: bool = True  # Add LoRA to MLP layers (including MoE)
    train_attn: bool = True  # Add LoRA to attention layers
```

### 2.2 Elixir Implementation

```elixir
# File: tinkex/lib/tinkex/types/lora_config.ex
defmodule Tinkex.Types.LoraConfig do
  @derive {Jason.Encoder, only: [:rank, :seed, :train_mlp, :train_attn, :train_unembed]}
  defstruct rank: 32,
            seed: nil,
            train_mlp: true,
            train_attn: true,
            train_unembed: true

  @type t :: %__MODULE__{
          rank: pos_integer(),
          seed: integer() | nil,
          train_mlp: boolean(),
          train_attn: boolean(),
          train_unembed: boolean()
        }
end

# File: crucible_kitchen/lib/crucible_kitchen/hyperparam.ex
defmodule CrucibleKitchen.Hyperparam do
  # Placeholder for hyperparameter utilities
end
```

### 2.3 Gaps in Elixir Implementation

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| LoRA config struct | `LoraConfig` | `Tinkex.Types.LoraConfig` | None |
| LR calculation | `get_lr()` | None | **GAP** |
| LoRA LR scaling | `get_lora_lr_over_full_finetune_lr()` | None | **GAP** |
| LoRA param count | `get_lora_param_count()` | None | **GAP** |
| Model hidden size lookup | `_get_hidden_size()` | None | **GAP** |

---

## 3. Quantization Patterns

### 3.1 How tinker-cookbook Handles Quantization

**Critical Insight**: Tinker does NOT support client-side quantization. The service runs models in their native precision on dedicated GPU infrastructure. Quantization (4-bit, 8-bit) is a LOCAL model loading concern that doesn't apply to Tinker's cloud architecture.

From `model-lineup.mdx`:

```markdown
## Available Models in Tinker

The table below shows the models that are currently available in Tinker.

| Model Name | Training Type | Architecture | Size |
|------------|---------------|--------------|------|
| Qwen/Qwen3-VL-235B-A22B-Instruct | Vision | MoE | Large |
| meta-llama/Llama-3.1-70B | Base | Dense | Large |
| ... | ... | ... | ... |

Note that the MoE models are much more cost effective than the dense models
as their cost is proportional to the number of active parameters.
```

### 3.2 What This Means for Elixir

The Elixir ecosystem does NOT need quantization support for Tinker integration. However, for local Nx/EXLA model loading (if ever implemented), quantization would be needed.

### 3.3 Future Considerations

If Bumblebee/Nx local model loading is needed:

| Feature | Status | Notes |
|---------|--------|-------|
| 4-bit quantization | Not needed for Tinker | Would need Nx/EXLA support |
| 8-bit quantization | Not needed for Tinker | Would need Nx/EXLA support |
| Mixed precision | Handled server-side | No client implementation needed |

---

## 4. Adapter Management (LoRA)

### 4.1 How tinker-cookbook Manages Adapters

LoRA adapters are created, saved, and loaded through the Tinker service.

#### Creating LoRA Training Client:

```python
# From training-sampling.mdx
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
```

#### Saving Weights for Sampling:

```python
# From save-load.mdx
# Save weights (creates a checkpoint path)
sampling_client = training_client.save_weights_and_get_sampling_client(
    name='pig-latin-model'
)
```

#### Downloading Checkpoints (CLI):

```bash
# From download-weights.mdx
tinker checkpoint download $TINKER_CHECKPOINT_PATH
```

#### Downloading Checkpoints (SDK):

```python
# From download-weights.mdx
import tinker
import urllib.request

sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(
    "tinker://<unique_id>/sampler_weights/final"
)
checkpoint_archive_url_response = future.result()

# Download the signed URL
urllib.request.urlretrieve(checkpoint_archive_url_response.url, "archive.tar")
```

#### Loading Public Weights:

```python
# From publish-weights.mdx
ckpt_path = "tinker://14bdf3a1-0b95-55c7-8659-5edb1bc870af/weights/checkpoint_id"
training_client = service_client.create_training_client_from_state(ckpt_path)
```

### 4.2 Elixir Implementation

```elixir
# File: tinkex/lib/tinkex/training_client.ex
defmodule Tinkex.TrainingClient do
  @spec save_weights_for_sampler(t(), keyword()) :: {:ok, Task.t()} | {:error, Error.t()}
  def save_weights_for_sampler(client, opts \\ []) do
    {:ok,
     Task.async(fn ->
       GenServer.call(client, {:save_weights_for_sampler, opts}, :infinity)
     end)}
  end

  @spec create_sampling_client_async(t(), String.t(), keyword()) :: Task.t()
  def create_sampling_client_async(client, model_path, opts \\ []) do
    Task.async(fn ->
      GenServer.call(client, {:create_sampling_client, model_path, opts}, :infinity)
    end)
  end
end

# File: tinkex/lib/tinkex/checkpoint_download.ex
defmodule Tinkex.CheckpointDownload do
  @spec download(RestClient.t(), String.t(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def download(rest_client, checkpoint_path, opts \\ []) do
    output_dir = Keyword.get(opts, :output_dir, File.cwd!())

    if String.starts_with?(checkpoint_path, "tinker://") do
      checkpoint_id = checkpoint_path
        |> String.replace("tinker://", "")
        |> String.replace("/", "_")

      target_path = Path.join(output_dir, checkpoint_id)

      with :ok <- check_target(target_path, force),
           {:ok, url_response} <-
             RestClient.get_checkpoint_archive_url(rest_client, checkpoint_path),
           {:ok, archive_path} <- download_archive(url_response.url, progress_fn),
           :ok <- extract_archive(archive_path, target_path) do
        File.rm(archive_path)
        {:ok, %{destination: target_path, checkpoint_path: checkpoint_path}}
      end
    else
      {:error, {:invalid_path, "Checkpoint path must start with 'tinker://'"}}
    end
  end
end
```

### 4.3 Gaps in Elixir Implementation

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| Create LoRA client | `create_lora_training_client` | `Tinkex.ServiceClient.create_lora_training_client` | None |
| Save weights | `save_weights_for_sampler` | `Tinkex.TrainingClient.save_weights_for_sampler` | None |
| Download checkpoint | CLI + SDK | `Tinkex.CheckpointDownload` | None |
| Load from checkpoint | `create_training_client_from_state` | Not implemented | **GAP** |
| Publish weights | CLI command | Not implemented | **GAP** |
| LoRA merging | Not in cookbook | N/A | N/A (server-side) |

---

## 5. Model Conversion and Export

### 5.1 How tinker-cookbook Handles Export

Checkpoints are exported as tar archives containing LoRA adapter weights:

```python
# Downloaded checkpoint structure:
# archive.tar
#   ├── adapter_config.json
#   ├── adapter_model.safetensors
#   └── ... (other LoRA files)
```

From `download-weights.mdx`:

> Replace `<unique_id>` with your Training Run ID. This will save the LoRA adapter weights and config inside the `archive.tar` file.

### 5.2 Checkpoint Archive Format

The exported format is compatible with HuggingFace PEFT library:

```json
// adapter_config.json
{
  "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
  "peft_type": "LORA",
  "r": 32,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "inference_mode": true
}
```

### 5.3 Elixir Implementation

```elixir
# Tinkex.CheckpointDownload handles extraction
defp extract_archive(archive_path, target_path) do
  File.mkdir_p!(target_path)

  case :erl_tar.extract(String.to_charlist(archive_path), [
         {:cwd, String.to_charlist(target_path)}
       ]) do
    :ok -> :ok
    {:error, reason} ->
      File.rm_rf(target_path)
      {:error, {:extraction_failed, reason}}
  end
end
```

### 5.4 Gaps in Elixir Implementation

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| Download checkpoint archive | Yes | Yes | None |
| Extract tar archive | Yes | Yes (`:erl_tar`) | None |
| Parse adapter_config.json | Not in cookbook | None | **GAP** |
| Convert to other formats | Not in cookbook | N/A | N/A |
| PEFT compatibility | Yes (implicit) | None | **GAP** |

---

## 6. Model Registry Patterns

### 6.1 How tinker-cookbook Uses Model Registry

The Tinker service maintains a registry of:
1. Supported base models
2. User's training runs (with checkpoints)

#### Querying Supported Models:

```python
# From training-sampling.mdx
service_client = tinker.ServiceClient()
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

Output:
```
- meta-llama/Llama-3.1-70B
- meta-llama/Llama-3.1-8B
- Qwen/Qwen3-VL-30B-A3B-Instruct
- Qwen/Qwen3-VL-235B-A22B-Instruct
...
```

#### Training Run Management:

```python
# From API types
class TrainingRun(BaseModel):
    training_run_id: str  # Unique identifier
    base_model: str  # Base model name
    model_owner: str  # Owner/creator
    is_lora: bool  # Whether using LoRA
    corrupted: bool  # Corruption state
    lora_rank: int | None  # LoRA rank if applicable
    last_request_time: datetime  # Last activity
    last_checkpoint: Checkpoint | None  # Most recent training checkpoint
    last_sampler_checkpoint: Checkpoint | None  # Most recent sampler checkpoint
    user_metadata: dict | None  # Custom metadata
```

#### Checkpoint Management:

```python
# From API types
class Checkpoint(BaseModel):
    checkpoint_id: str  # The checkpoint ID
    checkpoint_type: str  # "training" or "sampler"
    time: datetime  # Creation time
    tinker_path: str  # Full tinker:// path
    size_bytes: int  # Size in bytes
    public: bool  # Public accessibility
```

### 6.2 Static Model Information

```python
# From model_info.py
MODEL_INFO = {
    "meta-llama/Llama-3.1-8B": {
        "architecture": "llama",
        "hidden_size": 4096,
        "num_layers": 32,
        "renderer": "llama3",
    },
    "Qwen/Qwen3-30B-A3B": {
        "architecture": "qwen",
        "hidden_size": 4096,
        "num_layers": 64,
        "renderer": "qwen3",
    },
    # ...
}
```

### 6.3 Elixir Implementation

Currently minimal:

```elixir
# Tinkex has types but no registry queries implemented
defmodule Tinkex.Types.Checkpoint do
  @moduledoc """
  Checkpoint metadata structure.
  """
  defstruct [:checkpoint_id, :checkpoint_type, :time, :tinker_path, :size_bytes, :public]
end
```

### 6.4 Gaps in Elixir Implementation

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| Query supported models | `get_server_capabilities()` | None | **CRITICAL GAP** |
| List training runs | REST API | None | **GAP** |
| Checkpoint listing | `list_checkpoints()` | None | **GAP** |
| Static model info | `model_info.py` | None | **GAP** |
| Model architecture metadata | Yes | None | **GAP** |

---

## 7. Tokenizer Management

### 7.1 How tinker-cookbook Manages Tokenizers

#### Loading Tokenizers:

```python
# From tokenizer_utils.py
from transformers import AutoTokenizer

def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Get a tokenizer for a model.

    Special handling for Llama-3 models which require special access.
    """
    # Handle Llama-3 access issues
    if "Llama-3" in model_name:
        # Use public mirror tokenizer
        tokenizer_id = "baseten/Meta-Llama-3-tokenizer"
    else:
        tokenizer_id = model_name

    return AutoTokenizer.from_pretrained(tokenizer_id)

# Usage
tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

#### Integration with Training Client:

```python
# From training-sampling.mdx
tokenizer = training_client.get_tokenizer()

def process_example(example, tokenizer):
    prompt = f"English: {example['input']}\nPig Latin:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    # ...
```

### 7.2 Elixir Implementation

```elixir
# File: tinkex/lib/tinkex/tokenizer.ex
defmodule Tinkex.Tokenizer do
  @moduledoc """
  Tokenization entrypoint for the Tinkex SDK.
  """

  alias Tokenizers.{Encoding, Tokenizer}

  @llama3_tokenizer "baseten/Meta-Llama-3-tokenizer"
  @tokenizer_table :tinkex_tokenizers

  @spec get_tokenizer_id(String.t(), Tinkex.TrainingClient.t() | nil, keyword()) :: tokenizer_id()
  def get_tokenizer_id(model_name, training_client \\ nil, opts \\ []) do
    case fetch_tokenizer_id_from_client(training_client, opts) do
      {:ok, tokenizer_id} -> tokenizer_id
      _ -> apply_tokenizer_heuristics(model_name)
    end
  end

  @spec get_or_load_tokenizer(tokenizer_id(), keyword()) ::
          {:ok, Tokenizer.t()} | {:error, Error.t()}
  def get_or_load_tokenizer(tokenizer_id, opts \\ []) do
    ensure_table!()

    case :ets.lookup(@tokenizer_table, tokenizer_id) do
      [{^tokenizer_id, tokenizer}] ->
        {:ok, tokenizer}
      [] ->
        load_fun = Keyword.get(opts, :load_fun, &Tokenizer.from_pretrained/1)
        with {:ok, tokenizer} <- load_tokenizer(load_fun, tokenizer_id),
             {:ok, cached} <- cache_tokenizer(tokenizer_id, tokenizer) do
          {:ok, cached}
        end
    end
  end

  @spec encode(String.t(), tokenizer_id() | String.t(), keyword()) ::
          {:ok, [integer()]} | {:error, Error.t()}
  def encode(text, model_name, opts \\ []) do
    tokenizer_id = get_tokenizer_id(model_name, Keyword.get(opts, :training_client), opts)

    with {:ok, tokenizer} <- get_or_load_tokenizer(tokenizer_id, opts),
         {:ok, encoding} <- Tokenizer.encode(tokenizer, text) do
      {:ok, Encoding.get_ids(encoding)}
    end
  end

  @spec decode([integer()], tokenizer_id() | String.t(), keyword()) ::
          {:ok, String.t()} | {:error, Error.t()}
  def decode(ids, model_name, opts \\ []) do
    tokenizer_id = get_tokenizer_id(model_name, Keyword.get(opts, :training_client), opts)

    with {:ok, tokenizer} <- get_or_load_tokenizer(tokenizer_id, opts),
         {:ok, text} <- Tokenizer.decode(tokenizer, ids) do
      {:ok, text}
    end
  end

  defp apply_tokenizer_heuristics(model_name) do
    if String.contains?(model_name, "Llama-3") do
      @llama3_tokenizer
    else
      model_name
    end
  end
end
```

### 7.3 Tokenizer Helpers (Adapter-Specific)

There is no TokenizerClient port in crucible_train. Tokenizer access is adapter-specific:

- TrainingClient adapters may expose helper functions for tokenizer lookup.
- Dedicated helper modules (e.g., tokenizers-ex wrappers) can be used directly.

### 7.4 Gaps in Elixir Implementation

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| Load from HuggingFace | `AutoTokenizer.from_pretrained` | `Tokenizers.from_pretrained` | None |
| ETS caching | N/A (Python caching) | `Tinkex.Tokenizer` | None |
| Llama-3 workaround | Yes | Yes | None |
| Get from training client | `training_client.get_tokenizer()` | Adapter-specific helper | Minor gap |
| Special tokens handling | `add_special_tokens=True/False` | Not exposed | **GAP** |
| Chat template application | Via renderers | Not implemented | **GAP** |

---

## 8. Gap Analysis Summary

### 8.1 Critical Gaps (High Priority)

| Gap | Description | Impact | Recommended Action |
|-----|-------------|--------|-------------------|
| Query supported models | No `get_server_capabilities()` | Cannot discover available models | Add to `Tinkex.ServiceClient` |
| Model info constants | No static model metadata | Cannot calculate LR, param counts | Create `CrucibleKitchen.ModelInfo` |
| LR calculation utilities | No hyperparameter helpers | Must hardcode values | Port `hyperparam_utils.py` |
| Load from checkpoint | No `create_training_client_from_state` | Cannot resume training | Add to `Tinkex.ServiceClient` |

### 8.2 Medium Priority Gaps

| Gap | Description | Impact | Recommended Action |
|-----|-------------|--------|-------------------|
| List training runs | Cannot list user's runs | Limited run management | Add REST API integration |
| Checkpoint listing | Cannot list checkpoints | Cannot browse history | Add REST API integration |
| Publish weights | Cannot share models | Limited collaboration | Add CLI command |
| PEFT config parsing | Cannot read adapter configs | Limited checkpoint inspection | Create parser module |

### 8.3 Low Priority / Future Gaps

| Gap | Description | Impact | Recommended Action |
|-----|-------------|--------|-------------------|
| Chat templates | No Jinja-style templates | Must use renderers | Accept current approach |
| Local quantization | No 4/8-bit support | N/A for Tinker | Only if local Nx needed |
| Model architecture lookup | No hidden_size lookup | Affects LR calculation | Bundle with model_info |

### 8.4 Non-Gaps (Already Implemented)

- LoRA config struct
- Checkpoint download and extraction
- Basic tokenizer encode/decode with caching
- Training client creation
- Save weights for sampler
- Create sampling client from weights

---

## 9. Recommended Elixir Implementation Roadmap

### Phase 1: Core Model Management (Week 1-2)

1. **Add `Tinkex.ServiceClient.get_server_capabilities/1`**
   - Query available models from Tinker service
   - Parse into structured types

2. **Create `CrucibleKitchen.ModelInfo`**
   - Static model metadata (architecture, hidden_size, renderer)
   - Constants for common models

3. **Port `hyperparam_utils.py` to `CrucibleKitchen.Hyperparam`**
   - `get_lr/2` - calculate learning rate
   - `get_lora_lr_scaling/1` - LoRA LR multiplier
   - `get_lora_param_count/2` - trainable param estimate

### Phase 2: Checkpoint Management (Week 2-3)

4. **Add `Tinkex.ServiceClient.create_training_client_from_state/2`**
   - Resume training from checkpoint
   - Load optimizer state option

5. **Add `Tinkex.RestClient.list_training_runs/1`**
   - Query user's training runs
   - Pagination support

6. **Add `Tinkex.RestClient.list_checkpoints/2`**
   - List checkpoints for a training run
   - Filter by type (training/sampler)

### Phase 3: Enhanced Tokenization (Week 3-4)

7. **Improve `Tinkex.Tokenizer`**
   - Add `add_special_tokens` option
   - Expose `get_vocab_size/1`
   - Add batch encode/decode

8. **Create `CrucibleKitchen.Renderers.ChatTemplate`**
   - Apply chat templates to messages
   - Support Qwen3, Llama3, DeepSeek formats

### Phase 4: Developer Experience (Week 4)

9. **Add Tinkex CLI commands**
   - `mix tinkex.checkpoint.download`
   - `mix tinkex.checkpoint.publish`
   - `mix tinkex.runs.list`

10. **Create `CrucibleKitchen.CheckpointInspector`**
    - Parse adapter_config.json
    - Display checkpoint metadata
    - Validate PEFT compatibility

---

## Appendix A: File Reference

### Python tinker-cookbook Files Analyzed

| File | Purpose |
|------|---------|
| `model_info.py` | Static model metadata constants |
| `hyperparam_utils.py` | LR calculation, LoRA scaling |
| `tokenizer_utils.py` | Tokenizer loading utilities |
| `checkpoint_utils.py` | Checkpoint helpers |
| `renderers.py` | Message-to-token rendering |
| `completers.py` | Token/message completion interfaces |

### Elixir Files Referenced

| File | Purpose |
|------|---------|
| `tinkex/lib/tinkex/service_client.ex` | Service entry point |
| `tinkex/lib/tinkex/training_client.ex` | Training operations |
| `tinkex/lib/tinkex/tokenizer.ex` | Tokenization with caching |
| `tinkex/lib/tinkex/checkpoint_download.ex` | Checkpoint download/extract |
| `tinkex/lib/tinkex/types/lora_config.ex` | LoRA configuration struct |
| `hf_hub_ex/lib/hf_hub/download.ex` | HuggingFace file downloads |
| `hf_hub_ex/lib/hf_hub/api.ex` | HuggingFace API client |

---

## Appendix B: Code Examples

### Example: Recommended Model Info Module

```elixir
defmodule CrucibleKitchen.ModelInfo do
  @moduledoc """
  Static model information and metadata.

  Port of tinker_cookbook/model_info.py
  """

  # Model identifiers
  @llama3_1_8b "meta-llama/Llama-3.1-8B"
  @llama3_1_70b "meta-llama/Llama-3.1-70B"
  @qwen3_30b_a3b "Qwen/Qwen3-30B-A3B"

  @model_info %{
    @llama3_1_8b => %{
      architecture: :llama,
      hidden_size: 4096,
      num_layers: 32,
      renderer: :llama3
    },
    @llama3_1_70b => %{
      architecture: :llama,
      hidden_size: 8192,
      num_layers: 80,
      renderer: :llama3
    },
    @qwen3_30b_a3b => %{
      architecture: :qwen,
      hidden_size: 4096,
      num_layers: 64,
      renderer: :qwen3
    }
  }

  @spec get(String.t()) :: {:ok, map()} | {:error, :not_found}
  def get(model_name) do
    case Map.get(@model_info, model_name) do
      nil -> {:error, :not_found}
      info -> {:ok, info}
    end
  end

  @spec hidden_size(String.t()) :: {:ok, pos_integer()} | {:error, :not_found}
  def hidden_size(model_name) do
    case get(model_name) do
      {:ok, %{hidden_size: size}} -> {:ok, size}
      error -> error
    end
  end
end
```

### Example: Recommended Hyperparam Module

```elixir
defmodule CrucibleKitchen.Hyperparam do
  @moduledoc """
  Hyperparameter calculation utilities.

  Port of tinker_cookbook/hyperparam_utils.py
  """

  alias CrucibleKitchen.ModelInfo

  @base_lr 1.0e-5

  @spec get_lr(String.t(), pos_integer()) :: {:ok, float()} | {:error, term()}
  def get_lr(model_name, batch_size \\ 1) when batch_size > 0 do
    {:ok, @base_lr * :math.sqrt(batch_size)}
  end

  @spec get_lora_lr_scaling(String.t()) :: {:ok, float()} | {:error, term()}
  def get_lora_lr_scaling(model_name) do
    case ModelInfo.hidden_size(model_name) do
      {:ok, hidden_size} -> {:ok, 4 * :math.sqrt(hidden_size)}
      error -> error
    end
  end

  @spec get_lora_param_count(String.t(), pos_integer()) :: {:ok, pos_integer()} | {:error, term()}
  def get_lora_param_count(model_name, lora_rank \\ 32) do
    case ModelInfo.get(model_name) do
      {:ok, %{hidden_size: d, num_layers: n}} ->
        # Each layer has 2 attention matrices (q, k, v, o) and 3 MLP matrices
        # LoRA adds A (r x d) and B (d x r) to each
        matrices_per_layer = 7
        params_per_matrix = 2 * lora_rank * d
        {:ok, n * matrices_per_layer * params_per_matrix}
      error -> error
    end
  end
end
```

---

*Document generated: 2025-12-28*
*Analysis based on: tinker-cookbook (Python), tinkex, hf_hub_ex, crucible_kitchen (Elixir)*
