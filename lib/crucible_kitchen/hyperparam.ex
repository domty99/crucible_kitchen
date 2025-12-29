defmodule CrucibleKitchen.Hyperparam do
  @moduledoc """
  Hyperparameter calculation utilities for fine-tuning.

  This module provides intelligent defaults and calculations for training
  hyperparameters based on model architecture and training method.

  ## Key Functions

  - `get_lr/2` - Calculate optimal learning rate for a model
  - `get_lora_param_count/2` - Estimate trainable LoRA parameters
  - `get_hidden_size/1` - Get model hidden dimension

  ## Learning Rate Scaling

  LoRA training typically requires ~10x higher learning rate than full fine-tuning.
  The optimal LR also scales with model size - smaller models can use higher LRs.

  ## Examples

      # Get recommended LR for LoRA training
      lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", is_lora: true)
      # => 0.0002 (2e-4)

      # Get LR for full fine-tuning
      lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", is_lora: false)
      # => 0.00002 (2e-5)

      # Estimate LoRA parameters
      count = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B", rank: 32)
      # => ~20M parameters

  ## Model Support

  Optimized for:
  - Llama 3.x family (1B, 3B, 8B, 70B)
  - Qwen 3.x family
  - Mistral/Mixtral
  - DeepSeek

  Falls back to reasonable defaults for unknown models.
  """

  @type model_name :: String.t()

  # Known model hidden sizes (avoids needing to fetch config)
  @hidden_sizes %{
    "meta-llama/Llama-3.2-1B" => 2048,
    "meta-llama/Llama-3.2-1B-Instruct" => 2048,
    "meta-llama/Llama-3.2-3B" => 3072,
    "meta-llama/Llama-3.2-3B-Instruct" => 3072,
    "meta-llama/Llama-3.1-8B" => 4096,
    "meta-llama/Llama-3.1-8B-Instruct" => 4096,
    "meta-llama/Llama-3.1-70B" => 8192,
    "meta-llama/Llama-3.1-70B-Instruct" => 8192,
    "meta-llama/Llama-3.3-70B-Instruct" => 8192,
    "Qwen/Qwen3-0.5B" => 1024,
    "Qwen/Qwen3-1.5B" => 1536,
    "Qwen/Qwen3-4B" => 2560,
    "Qwen/Qwen3-8B" => 4096,
    "Qwen/Qwen3-14B" => 5120,
    "Qwen/Qwen3-32B" => 5120,
    "Qwen/Qwen3-72B" => 8192,
    "mistralai/Mistral-7B-v0.1" => 4096,
    "mistralai/Mistral-7B-Instruct-v0.2" => 4096,
    "mistralai/Mixtral-8x7B-v0.1" => 4096,
    "deepseek-ai/DeepSeek-V2" => 5120,
    "moonshotai/Kimi-K2-Thinking" => 7168
  }

  # Default hidden sizes by model family pattern
  @family_defaults %{
    "llama-3.2-1b" => 2048,
    "llama-3.2-3b" => 3072,
    "llama-3.1-8b" => 4096,
    "llama-3.1-70b" => 8192,
    "llama-3.3-70b" => 8192,
    "qwen3-0.5b" => 1024,
    "qwen3-1.5b" => 1536,
    "qwen3-4b" => 2560,
    "qwen3-8b" => 4096,
    "qwen3-14b" => 5120,
    "qwen3-32b" => 5120,
    "qwen3-72b" => 8192,
    "mistral-7b" => 4096,
    "mixtral-8x7b" => 4096
  }

  @doc """
  Get the optimal learning rate for a model.

  ## Parameters

  - `model_name` - Model identifier (e.g., "meta-llama/Llama-3.1-8B")
  - `opts` - Options:
    - `:is_lora` - Whether using LoRA (default: true). LoRA uses ~10x higher LR.
    - `:base_lr` - Base learning rate (default: 5e-5)

  ## Returns

  The recommended learning rate as a float.

  ## Examples

      Hyperparam.get_lr("meta-llama/Llama-3.1-8B")
      # => 0.0002 (for LoRA)

      Hyperparam.get_lr("meta-llama/Llama-3.1-8B", is_lora: false)
      # => 0.00002 (for full fine-tuning)

  ## Formula

  The LR is scaled based on model hidden size:
  - Base LR = 5e-5
  - LoRA multiplier = 10x
  - Size scaling = (2000 / hidden_size) ^ exponent

  Where exponent varies by model family (empirically determined).
  """
  @spec get_lr(model_name(), keyword()) :: float()
  def get_lr(model_name, opts \\ []) do
    is_lora = Keyword.get(opts, :is_lora, true)
    base_lr = Keyword.get(opts, :base_lr, 5.0e-5)

    lora_multiplier = if is_lora, do: 10.0, else: 1.0
    lr = base_lr * lora_multiplier

    hidden_size = get_hidden_size(model_name)
    exponent = get_model_exponent(model_name)

    lr * :math.pow(2000 / hidden_size, exponent)
  end

  @doc """
  Get the ratio of LoRA LR to full fine-tuning LR.

  Currently returns a fixed 10x multiplier based on empirical findings.
  See "LoRA Without Regret" for details.
  """
  @spec get_lora_lr_over_full_finetune_lr(model_name(), keyword()) :: float()
  def get_lora_lr_over_full_finetune_lr(_model_name, _opts \\ []) do
    10.0
  end

  @doc """
  Get the hidden size (embedding dimension) for a model.

  ## Parameters

  - `model_name` - Model identifier

  ## Returns

  The hidden dimension as an integer.

  Falls back to 4096 for unknown models.
  """
  @spec get_hidden_size(model_name()) :: pos_integer()
  def get_hidden_size(model_name) do
    # First check exact match
    case Map.get(@hidden_sizes, model_name) do
      nil -> infer_hidden_size(model_name)
      size -> size
    end
  end

  @doc """
  Estimate the number of trainable LoRA parameters.

  ## Parameters

  - `model_name` - Model identifier
  - `opts` - Options:
    - `:rank` - LoRA rank (default: 32)
    - `:include_experts` - Include MoE expert layers (default: true)

  ## Returns

  Estimated parameter count as an integer.

  ## Formula

  For each linear layer with shape [out_dim, in_dim]:
  - LoRA adds rank * (out_dim + in_dim) parameters
  - Total = rank * sum(out_dim + in_dim for each layer)

  Note: This is an estimate based on typical model architectures.
  """
  @spec get_lora_param_count(model_name(), keyword()) :: pos_integer()
  def get_lora_param_count(model_name, opts \\ []) do
    rank = Keyword.get(opts, :rank, 32)
    hidden_size = get_hidden_size(model_name)

    # Estimate based on typical transformer architecture
    # Each layer has: q, k, v, o projections + 2-3 FFN projections
    # Assuming hidden_size and 4x intermediate size
    intermediate_size = hidden_size * 4
    num_layers = estimate_num_layers(model_name, hidden_size)

    # Per layer: q, k, v, o = 4 * (hidden + hidden)
    #            ffn up/down = 2 * (hidden + intermediate)
    #            ffn gate (if SwiGLU) = hidden + intermediate
    attention_params = 4 * 2 * hidden_size
    ffn_params = 3 * (hidden_size + intermediate_size)

    per_layer = attention_params + ffn_params
    total_dim_sum = per_layer * num_layers

    rank * total_dim_sum
  end

  @doc """
  Get the LR multiplier for a specific model (for cross-model LR transfer).

  If you have an optimal LR for model A, you can estimate optimal LR for model B:
  LR_B = LR_A * get_lora_lr_multiplier(B) / get_lora_lr_multiplier(A)
  """
  @spec get_lora_lr_multiplier(model_name()) :: float()
  def get_lora_lr_multiplier(model_name) do
    # Based on: multiplier ~ 1/sqrt(param_count) * lora_factor
    param_count = estimate_full_param_count(model_name)
    lora_factor = get_lora_lr_over_full_finetune_lr(model_name)

    1.0 / :math.sqrt(param_count) * lora_factor
  end

  @doc """
  Get recommended batch size for a model based on typical GPU memory.

  ## Parameters

  - `model_name` - Model identifier
  - `opts` - Options:
    - `:gpu_memory_gb` - Available GPU memory in GB (default: 24)
    - `:is_lora` - Whether using LoRA (default: true)
    - `:sequence_length` - Max sequence length (default: 2048)

  ## Returns

  Recommended batch size as an integer.
  """
  @spec get_recommended_batch_size(model_name(), keyword()) :: pos_integer()
  def get_recommended_batch_size(model_name, opts \\ []) do
    gpu_memory_gb = Keyword.get(opts, :gpu_memory_gb, 24)
    is_lora = Keyword.get(opts, :is_lora, true)
    seq_len = Keyword.get(opts, :sequence_length, 2048)

    # Rough heuristics based on model size and memory
    hidden_size = get_hidden_size(model_name)

    # Larger models need more memory per sample
    # LoRA reduces memory significantly
    lora_factor = if is_lora, do: 4.0, else: 1.0
    seq_factor = 2048 / seq_len

    base_batch =
      cond do
        hidden_size <= 2048 -> 32
        hidden_size <= 4096 -> 16
        hidden_size <= 5120 -> 8
        true -> 4
      end

    memory_factor = gpu_memory_gb / 24.0

    max(1, round(base_batch * lora_factor * seq_factor * memory_factor))
  end

  @doc """
  Get the model family for a model name.

  Returns one of: :llama, :qwen, :mistral, :deepseek, :unknown
  """
  @spec get_model_family(model_name()) :: atom()
  def get_model_family(model_name) do
    lower = String.downcase(model_name)

    cond do
      String.contains?(lower, "llama") -> :llama
      String.contains?(lower, "qwen") -> :qwen
      String.contains?(lower, "mistral") or String.contains?(lower, "mixtral") -> :mistral
      String.contains?(lower, "deepseek") -> :deepseek
      String.contains?(lower, "kimi") -> :kimi
      true -> :unknown
    end
  end

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp get_model_exponent(model_name) do
    case get_model_family(model_name) do
      :llama -> 0.781
      :qwen -> 0.0775
      :kimi -> 0.0775
      :mistral -> 0.5
      :deepseek -> 0.5
      :unknown -> 0.5
    end
  end

  defp infer_hidden_size(model_name) do
    lower = String.downcase(model_name)

    # Check family patterns
    result =
      Enum.find_value(@family_defaults, fn {pattern, size} ->
        if String.contains?(lower, pattern), do: size
      end)

    case result do
      nil -> infer_from_size_suffix(lower)
      size -> size
    end
  end

  @size_suffix_patterns [
    {~w(70b 72b), 8192},
    {~w(32b 34b), 5120},
    {~w(14b 13b), 5120},
    {~w(8b 7b), 4096},
    {~w(3b 4b), 3072},
    {~w(1b 1.5b), 2048},
    {["0.5b"], 1024}
  ]

  defp infer_from_size_suffix(lower) do
    Enum.find_value(@size_suffix_patterns, 4096, fn {patterns, size} ->
      if Enum.any?(patterns, &String.contains?(lower, &1)), do: size
    end)
  end

  @layer_map %{
    {:llama, 2048} => 16,
    {:llama, 3072} => 28,
    {:llama, 4096} => 32,
    {:llama, 8192} => 80,
    {:qwen, 1024} => 24,
    {:qwen, 1536} => 28,
    {:qwen, 2560} => 40,
    {:qwen, 4096} => 32,
    {:qwen, 5120} => 48,
    {:qwen, 8192} => 80,
    {:mistral, 4096} => 32
  }

  defp estimate_num_layers(model_name, hidden_size) do
    # Estimate layers based on hidden size and model family
    family = get_model_family(model_name)
    Map.get(@layer_map, {family, hidden_size}, max(16, div(hidden_size, 128)))
  end

  @param_count_patterns [
    {~w(70b 72b), 70.0e9},
    {~w(32b 34b), 32.0e9},
    {~w(14b 13b), 14.0e9},
    {~w(8b 7b), 8.0e9},
    {~w(3b 4b), 3.0e9},
    {~w(1b 1.5b), 1.5e9},
    {["0.5b"], 0.5e9}
  ]

  defp estimate_full_param_count(model_name) do
    lower = String.downcase(model_name)

    Enum.find_value(@param_count_patterns, 7.0e9, fn {patterns, count} ->
      if Enum.any?(patterns, &String.contains?(lower, &1)), do: count
    end)
  end
end
