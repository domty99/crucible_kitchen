defmodule CrucibleKitchen.Stages.DPOStagesTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Context

  alias CrucibleKitchen.Stages.{
    BuildPreferenceDataset,
    ComputeReferenceLogprobs,
    DPOForwardBackward,
    GetPreferenceBatch,
    LogDPOMetrics
  }

  describe "BuildPreferenceDataset" do
    test "name returns :build_preference_dataset" do
      assert BuildPreferenceDataset.name() == :build_preference_dataset
    end

    test "builds preference dataset from raw data" do
      raw_dataset = [
        %{"prompt" => "Question 1", "chosen" => "Good answer", "rejected" => "Bad answer"},
        %{"prompt" => "Question 2", "chosen" => "Better", "rejected" => "Worse"}
      ]

      context =
        build_context(%{batch_size: 1})
        |> Context.put_state(:raw_dataset, raw_dataset)

      assert {:ok, result} = BuildPreferenceDataset.execute(context)
      assert result.state.preference_dataset != nil
      assert result.state.preference_dataset.num_pairs == 2
      assert result.state.preference_dataset.num_batches == 2
    end

    test "filters invalid pairs" do
      raw_dataset = [
        %{"prompt" => "Valid", "chosen" => "Yes", "rejected" => "No"},
        # Invalid - empty prompt
        %{"prompt" => "", "chosen" => "Yes", "rejected" => "No"},
        # Invalid - empty chosen
        %{"prompt" => "Valid", "chosen" => "", "rejected" => "No"}
      ]

      context =
        build_context(%{batch_size: 10})
        |> Context.put_state(:raw_dataset, raw_dataset)

      assert {:ok, result} = BuildPreferenceDataset.execute(context)
      assert result.state.num_preference_pairs == 1
    end

    test "validation requires raw_dataset" do
      context = build_context(%{})
      assert {:error, _} = BuildPreferenceDataset.validate(context)
    end
  end

  describe "GetPreferenceBatch" do
    test "name returns :get_preference_batch" do
      assert GetPreferenceBatch.name() == :get_preference_batch
    end

    test "gets batch from preference dataset" do
      pairs = [
        %{prompt: "Q1", chosen: "A1", rejected: "B1"},
        %{prompt: "Q2", chosen: "A2", rejected: "B2"}
      ]

      context =
        build_context(%{})
        |> Context.put_state(:preference_dataset, %{
          pairs: pairs,
          batch_size: 1,
          num_batches: 2,
          num_pairs: 2
        })
        |> Context.put_state(:pref_batches_index, 0)

      assert {:ok, result} = GetPreferenceBatch.execute(context)
      assert length(result.state.preference_batch) == 1
      assert result.state.batch_index == 0
    end
  end

  describe "ComputeReferenceLogprobs" do
    test "name returns :compute_reference_logprobs" do
      assert ComputeReferenceLogprobs.name() == :compute_reference_logprobs
    end

    test "computes mock logprobs when no adapter" do
      batch = [
        %{prompt: "Q1", chosen: "A1", rejected: "B1"},
        %{prompt: "Q2", chosen: "A2", rejected: "B2"}
      ]

      context =
        build_context_without_training(%{})
        |> Context.put_state(:preference_batch, batch)

      assert {:ok, result} = ComputeReferenceLogprobs.execute(context)
      assert length(result.state.ref_chosen_logprobs) == 2
      assert length(result.state.ref_rejected_logprobs) == 2
    end

    test "validation requires preference_batch" do
      context = build_context(%{})
      assert {:error, _} = ComputeReferenceLogprobs.validate(context)
    end
  end

  describe "DPOForwardBackward" do
    test "name returns :dpo_forward_backward" do
      assert DPOForwardBackward.name() == :dpo_forward_backward
    end

    test "executes mock DPO when no adapter" do
      batch = [%{prompt: "Q", chosen: "A", rejected: "B"}]

      context =
        build_context_without_training(%{dpo_beta: 0.1})
        |> Context.put_state(:preference_batch, batch)
        |> Context.put_state(:ref_chosen_logprobs, [-1.0])
        |> Context.put_state(:ref_rejected_logprobs, [-2.0])

      assert {:ok, result} = DPOForwardBackward.execute(context)
      assert result.state.dpo_metrics != nil
      assert result.state.dpo_metrics.beta == 0.1
    end

    test "validation requires reference logprobs" do
      context =
        build_context(%{})
        |> Context.put_state(:preference_batch, [])

      assert {:error, msg} = DPOForwardBackward.validate(context)
      assert String.contains?(msg, "ref_chosen_logprobs")
    end
  end

  describe "LogDPOMetrics" do
    test "name returns :log_dpo_metrics" do
      assert LogDPOMetrics.name() == :log_dpo_metrics
    end

    test "logs metrics and increments step" do
      context =
        build_context(%{})
        |> Context.put_state(:dpo_metrics, %{
          loss: 0.5,
          accuracy: 0.7,
          chosen_reward: 1.0,
          rejected_reward: -1.0,
          margin: 2.0
        })
        |> Context.put_state(:global_step, 10)

      assert {:ok, result} = LogDPOMetrics.execute(context)
      assert result.state.global_step == 11

      # Check metrics recorded
      assert Enum.any?(result.metrics, &(&1.name == :dpo_loss))
      assert Enum.any?(result.metrics, &(&1.name == :dpo_accuracy))
    end

    test "emits telemetry event" do
      :telemetry.attach(
        "log-dpo-metrics-test",
        [:crucible_kitchen, :dpo, :step],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      context =
        build_context(%{})
        |> Context.put_state(:dpo_metrics, %{loss: 0.3, accuracy: 0.8})
        |> Context.put_state(:global_step, 5)

      {:ok, _} = LogDPOMetrics.execute(context)

      assert_receive {:telemetry, [:crucible_kitchen, :dpo, :step], measurements, _}
      assert measurements.loss == 0.3

      :telemetry.detach("log-dpo-metrics-test")
    end
  end

  defp build_context(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        training_client: CrucibleKitchen.Adapters.Noop.TrainingClient,
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end

  defp build_context_without_training(extra_config) do
    config = Map.merge(%{}, extra_config)

    Context.new(
      config,
      %{
        dataset_store: CrucibleKitchen.Adapters.Noop.DatasetStore
      }
    )
  end
end
