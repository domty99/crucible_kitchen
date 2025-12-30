defmodule CrucibleKitchen.Adapters.Noop.FeedbackClientTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.FeedbackClient

  describe "check_triggers/2" do
    test "returns empty list when no triggers are configured and thresholds not exceeded" do
      opts = [drift_score: 0.1, quality_average: 0.85, event_count: 500]

      triggers = FeedbackClient.check_triggers(opts, "test-deployment")

      assert triggers == []
    end

    test "returns configured triggers directly" do
      opts = [triggers: [{:trigger, :drift_threshold}, {:trigger, :quality_drop}]]

      triggers = FeedbackClient.check_triggers(opts, "test-deployment")

      assert triggers == [{:trigger, :drift_threshold}, {:trigger, :quality_drop}]
    end

    test "triggers drift_threshold when score exceeds threshold" do
      opts = [drift_score: 0.3, drift_threshold: 0.2]

      triggers = FeedbackClient.check_triggers(opts, "test-deployment")

      assert {:trigger, :drift_threshold} in triggers
    end

    test "triggers quality_drop when average below threshold" do
      opts = [quality_average: 0.6, quality_threshold: 0.7]

      triggers = FeedbackClient.check_triggers(opts, "test-deployment")

      assert {:trigger, :quality_drop} in triggers
    end

    test "triggers data_count when count meets threshold" do
      opts = [event_count: 1500, data_count_threshold: 1000]

      triggers = FeedbackClient.check_triggers(opts, "test-deployment")

      assert {:trigger, :data_count} in triggers
    end

    test "can trigger multiple conditions" do
      opts = [
        drift_score: 0.5,
        drift_threshold: 0.2,
        quality_average: 0.5,
        quality_threshold: 0.7,
        event_count: 2000,
        data_count_threshold: 1000
      ]

      triggers = FeedbackClient.check_triggers(opts, "test-deployment")

      assert {:trigger, :drift_threshold} in triggers
      assert {:trigger, :quality_drop} in triggers
      assert {:trigger, :data_count} in triggers
    end
  end

  describe "curate/2" do
    test "returns configured number of examples" do
      opts = [curated_count: 50]

      assert {:ok, examples} = FeedbackClient.curate(opts, "test-deployment")
      assert length(examples) == 50
    end

    test "defaults to 100 examples" do
      assert {:ok, examples} = FeedbackClient.curate([], "test-deployment")
      assert length(examples) == 100
    end

    test "includes expected fields in each example" do
      assert {:ok, [example | _]} = FeedbackClient.curate([], "test-deployment")

      assert Map.has_key?(example, :inference_event_id)
      assert Map.has_key?(example, :deployment_id)
      assert Map.has_key?(example, :curation_source)
      assert Map.has_key?(example, :curation_score)
      assert Map.has_key?(example, :prompt)
      assert Map.has_key?(example, :response)
    end

    test "assigns curation sources from expected set" do
      assert {:ok, examples} = FeedbackClient.curate([curated_count: 20], "test")

      sources = Enum.map(examples, & &1.curation_source) |> Enum.uniq()

      for source <- sources do
        assert source in [:user_edit, :high_quality, :hard_example, :diverse]
      end
    end
  end

  describe "export/2" do
    test "returns path for jsonl format" do
      opts = [format: :jsonl]

      assert {:ok, path} = FeedbackClient.export(opts, "deploy-123")
      assert String.ends_with?(path, "_deploy-123.jsonl")
    end

    test "returns path for huggingface format" do
      opts = [format: :huggingface]

      assert {:ok, path} = FeedbackClient.export(opts, "deploy-123")
      assert String.ends_with?(path, "_deploy-123_hf")
    end

    test "returns path for parquet format" do
      opts = [format: :parquet]

      assert {:ok, path} = FeedbackClient.export(opts, "deploy-123")
      assert String.ends_with?(path, "_deploy-123.parquet")
    end

    test "uses custom export path when provided" do
      opts = [export_path: "/custom/path", format: :jsonl]

      assert {:ok, path} = FeedbackClient.export(opts, "deploy-123")
      assert path == "/custom/path_deploy-123.jsonl"
    end
  end

  describe "export_preference_pairs/2" do
    test "returns preference pairs path" do
      assert {:ok, path} = FeedbackClient.export_preference_pairs([], "deploy-123")
      assert String.contains?(path, "deploy-123")
      assert String.ends_with?(path, ".jsonl")
    end
  end

  describe "update_baseline/2" do
    test "returns :ok" do
      assert :ok = FeedbackClient.update_baseline([], "test-deployment")
    end
  end

  describe "get_drift_status/2" do
    test "returns drift scores" do
      opts = [drift_score: 0.15]

      assert {:ok, status} = FeedbackClient.get_drift_status(opts, "test")
      assert status.input_drift == 0.15
      assert status.output_drift == 0.15 * 0.8
    end
  end

  describe "get_quality_metrics/2" do
    test "returns quality metrics" do
      opts = [quality_average: 0.9, event_count: 1000]

      assert {:ok, metrics} = FeedbackClient.get_quality_metrics(opts, "test")
      assert metrics.average == 0.9
      assert metrics.count == 1000
    end
  end
end
