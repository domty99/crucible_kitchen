defmodule CrucibleKitchen.Adapters.Noop.TrainingClient do
  @moduledoc """
  Noop adapter for TrainingClient port.

  This adapter does nothing but returns successful responses. Useful for:
  - Testing stages in isolation
  - Dry-run mode
  - Development without backend access
  """

  @behaviour CrucibleTrain.Ports.TrainingClient

  @impl true
  def start_session(_opts, config) do
    session = %{
      id: generate_id(),
      model: config[:model] || "noop-model",
      started_at: DateTime.utc_now(),
      step: 0
    }

    {:ok, session}
  end

  @impl true
  def forward_backward(_opts, session, datums) do
    batch_size = length(datums)

    future = %{
      type: :forward_backward,
      session_id: session.id,
      batch_size: batch_size,
      submitted_at: DateTime.utc_now()
    }

    future
  end

  @impl true
  def optim_step(_opts, session, learning_rate) do
    future = %{
      type: :optim_step,
      session_id: session.id,
      learning_rate: learning_rate,
      submitted_at: DateTime.utc_now()
    }

    future
  end

  @impl true
  def await(_opts, future) do
    # Simulate some realistic metrics
    result =
      case future.type do
        :forward_backward ->
          %{
            loss: :rand.uniform() * 2,
            tokens_processed: Map.get(future, :batch_size, 1) * 512,
            duration_ms: :rand.uniform(100) + 50
          }

        :optim_step ->
          %{
            grad_norm: :rand.uniform() * 10,
            duration_ms: :rand.uniform(50) + 10
          }

        :dpo_forward_backward ->
          %{
            loss: :rand.uniform() * 0.5,
            chosen_reward: :rand.uniform() * 2,
            rejected_reward: :rand.uniform() * 2 - 1,
            duration_ms: :rand.uniform(100) + 50
          }

        :distillation_forward_backward ->
          %{
            total_loss: :rand.uniform() * 2,
            kl_loss: :rand.uniform(),
            ce_loss: :rand.uniform(),
            duration_ms: :rand.uniform(100) + 50
          }

        _ ->
          %{}
      end

    {:ok, result}
  end

  @impl true
  def save_checkpoint(_opts, _session, _name) do
    :ok
  end

  @impl true
  def load_checkpoint(_opts, _session, _path) do
    :ok
  end

  @impl true
  def close_session(_opts, _session) do
    :ok
  end

  def get_tokenizer(_opts, _session) do
    # Return a simple tokenizer-like structure
    tokenizer = %{
      type: :noop,
      vocab_size: 32_000,
      pad_token_id: 0,
      eos_token_id: 1,
      bos_token_id: 2
    }

    {:ok, tokenizer}
  end

  # ============================================================================
  # RL-specific callbacks
  # ============================================================================

  def do_rollout(_opts, _session, prompts) do
    # Generate mock trajectories
    trajectories =
      Enum.map(prompts, fn _prompt ->
        steps = :rand.uniform(10) + 5

        %{
          observations: Enum.map(1..steps, fn _ -> :rand.uniform(1000) end),
          actions: Enum.map(1..steps, fn _ -> :rand.uniform(100) end),
          rewards: Enum.map(1..steps, fn _ -> :rand.uniform() * 2 - 1 end),
          values: Enum.map(1..steps, fn _ -> :rand.uniform() end),
          log_probs: Enum.map(1..steps, fn _ -> -:rand.uniform() * 5 end)
        }
      end)

    {:ok, trajectories}
  end

  def ppo_update(_opts, _session, _batch, _opts_kw) do
    result = %{
      policy_loss: :rand.uniform() * 0.5,
      value_loss: :rand.uniform() * 0.3,
      entropy: :rand.uniform() * 0.1,
      kl_divergence: :rand.uniform() * 0.05
    }

    {:ok, result}
  end

  # ============================================================================
  # DPO-specific callbacks
  # ============================================================================

  def compute_reference_logprobs(_opts, _session, batch) do
    ref_logprobs =
      Enum.with_index(batch)
      |> Enum.map(fn {_pair, idx} ->
        {idx, %{chosen: -:rand.uniform() * 5, rejected: -:rand.uniform() * 5}}
      end)
      |> Map.new()

    {:ok, ref_logprobs}
  end

  def dpo_forward_backward(_opts, session, _batch, _ref_logprobs, _opts_kw) do
    future = %{
      type: :dpo_forward_backward,
      session_id: session.id,
      submitted_at: DateTime.utc_now()
    }

    {:ok, future}
  end

  # ============================================================================
  # Distillation-specific callbacks
  # ============================================================================

  def load_teacher(_opts, teacher_model) do
    teacher = %{
      id: generate_id(),
      model: teacher_model,
      loaded_at: DateTime.utc_now()
    }

    {:ok, teacher}
  end

  def teacher_inference(_opts, _teacher, batch, _opts_kw) do
    # Generate mock logits
    logits = Enum.map(batch, fn _sample -> generate_mock_logits() end)
    {:ok, logits}
  end

  defp generate_mock_logits do
    # Simulate logits for 1000 vocab size, 128 sequence length
    Enum.map(1..128, fn _ -> generate_vocab_logits() end)
  end

  defp generate_vocab_logits do
    Enum.map(1..1000, fn _ -> :rand.uniform() * 10 - 5 end)
  end

  def distillation_forward_backward(_opts, session, _batch, _teacher_logits, _opts_kw) do
    future = %{
      type: :distillation_forward_backward,
      session_id: session.id,
      submitted_at: DateTime.utc_now()
    }

    {:ok, future}
  end

  def unload_teacher(_opts, _teacher) do
    :ok
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
