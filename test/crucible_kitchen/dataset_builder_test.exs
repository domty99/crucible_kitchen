defmodule CrucibleKitchen.DatasetBuilderTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.DatasetBuilder
  alias CrucibleKitchen.Renderers.Message

  describe "from_list/1" do
    test "creates dataset from list" do
      data = [%{"a" => 1}, %{"a" => 2}]
      {:ok, dataset} = DatasetBuilder.from_list(data)

      assert dataset == data
    end
  end

  describe "map/2" do
    test "transforms each row" do
      {:ok, dataset} = DatasetBuilder.from_list([%{"x" => 1}, %{"x" => 2}])

      result = DatasetBuilder.map(dataset, fn row -> Map.put(row, "y", row["x"] * 2) end)

      assert Enum.at(result, 0)["y"] == 2
      assert Enum.at(result, 1)["y"] == 4
    end
  end

  describe "filter/2" do
    test "filters rows by predicate" do
      {:ok, dataset} = DatasetBuilder.from_list([%{"x" => 1}, %{"x" => 2}, %{"x" => 3}])

      result = DatasetBuilder.filter(dataset, fn row -> row["x"] > 1 end)

      assert length(result) == 2
      assert Enum.all?(result, fn row -> row["x"] > 1 end)
    end
  end

  describe "select/2" do
    test "selects specific fields" do
      {:ok, dataset} =
        DatasetBuilder.from_list([%{"a" => 1, "b" => 2, "c" => 3}])

      result = DatasetBuilder.select(dataset, ["a", "c"])

      assert Enum.at(result, 0) == %{"a" => 1, "c" => 3}
    end
  end

  describe "rename/2" do
    test "renames fields" do
      {:ok, dataset} = DatasetBuilder.from_list([%{"old_name" => "value"}])

      result = DatasetBuilder.rename(dataset, %{"old_name" => "new_name"})

      assert Enum.at(result, 0)["new_name"] == "value"
      refute Map.has_key?(Enum.at(result, 0), "old_name")
    end
  end

  describe "shuffle/2" do
    test "shuffles dataset" do
      {:ok, dataset} =
        DatasetBuilder.from_list(Enum.map(1..100, fn i -> %{"i" => i} end))

      result = DatasetBuilder.shuffle(dataset, 42)

      # Should have same elements
      assert length(result) == length(dataset)
      # Order should be different (very unlikely to be same)
      assert result != dataset
    end

    test "deterministic with seed" do
      {:ok, dataset} =
        DatasetBuilder.from_list(Enum.map(1..10, fn i -> %{"i" => i} end))

      result1 = DatasetBuilder.shuffle(dataset, 42)
      result2 = DatasetBuilder.shuffle(dataset, 42)

      assert result1 == result2
    end
  end

  describe "take/2" do
    test "takes first n rows" do
      {:ok, dataset} =
        DatasetBuilder.from_list([%{"x" => 1}, %{"x" => 2}, %{"x" => 3}])

      result = DatasetBuilder.take(dataset, 2)

      assert length(result) == 2
      assert Enum.at(result, 0)["x"] == 1
    end
  end

  describe "sample/3" do
    test "samples n random rows" do
      {:ok, dataset} =
        DatasetBuilder.from_list(Enum.map(1..100, fn i -> %{"i" => i} end))

      result = DatasetBuilder.sample(dataset, 10, 42)

      assert length(result) == 10
    end
  end

  describe "split/3" do
    test "splits into train/val/test" do
      {:ok, dataset} =
        DatasetBuilder.from_list(Enum.map(1..100, fn i -> %{"i" => i} end))

      splits = DatasetBuilder.split(dataset, {0.8, 0.1, 0.1}, 42)

      assert length(splits.train) == 80
      assert length(splits.val) == 10
      assert length(splits.test) == 10

      # No overlap
      all_ids = Enum.map(splits.train ++ splits.val ++ splits.test, & &1["i"])
      assert length(Enum.uniq(all_ids)) == 100
    end

    test "uses default ratios" do
      {:ok, dataset} =
        DatasetBuilder.from_list(Enum.map(1..100, fn i -> %{"i" => i} end))

      splits = DatasetBuilder.split(dataset)

      assert length(splits.train) == 80
    end
  end

  describe "to_chat/2" do
    test "transforms rows to chat format" do
      {:ok, dataset} =
        DatasetBuilder.from_list([
          %{"question" => "What is 2+2?", "answer" => "4"}
        ])

      result =
        DatasetBuilder.to_chat(dataset, fn row ->
          [
            Message.user(row["question"]),
            Message.assistant(row["answer"])
          ]
        end)

      messages = Enum.at(result, 0)["messages"]
      assert length(messages) == 2
      assert Enum.at(messages, 0).role == "user"
      assert Enum.at(messages, 1).role == "assistant"
    end
  end

  describe "instruction_chat_format/2" do
    test "creates instruction format messages" do
      row = %{
        "instruction" => "Translate to French",
        "input" => "Hello",
        "output" => "Bonjour"
      }

      messages = DatasetBuilder.instruction_chat_format(row)

      assert length(messages) == 2
      assert Enum.at(messages, 0).role == "user"
      assert String.contains?(Enum.at(messages, 0).content, "Translate to French")
      assert String.contains?(Enum.at(messages, 0).content, "Hello")
      assert Enum.at(messages, 1).content == "Bonjour"
    end

    test "includes system prompt when provided" do
      row = %{"instruction" => "Help me", "output" => "Sure!"}

      messages = DatasetBuilder.instruction_chat_format(row, system_prompt: "You are helpful.")

      assert length(messages) == 3
      assert Enum.at(messages, 0).role == "system"
    end

    test "handles response key instead of output" do
      row = %{"instruction" => "Question", "response" => "Answer"}

      messages = DatasetBuilder.instruction_chat_format(row)

      assert Enum.at(messages, 1).content == "Answer"
    end
  end

  describe "qa_chat_format/2" do
    test "creates QA format messages" do
      row = %{"question" => "What is 2+2?", "answer" => "4"}

      messages = DatasetBuilder.qa_chat_format(row)

      assert length(messages) == 2
      assert Enum.at(messages, 0).content == "What is 2+2?"
      assert Enum.at(messages, 1).content == "4"
    end

    test "includes context when present" do
      row = %{
        "context" => "Mathematics is the study of numbers.",
        "question" => "What is math?",
        "answer" => "The study of numbers."
      }

      messages = DatasetBuilder.qa_chat_format(row)

      assert String.contains?(Enum.at(messages, 0).content, "Context:")
      assert String.contains?(Enum.at(messages, 0).content, "Mathematics")
    end
  end

  describe "stats/1" do
    test "returns dataset statistics" do
      {:ok, dataset} =
        DatasetBuilder.from_list([
          %{"a" => 1, "b" => 2},
          %{"a" => 3, "b" => 4}
        ])

      stats = DatasetBuilder.stats(dataset)

      assert stats.count == 2
      assert "a" in stats.fields
      assert "b" in stats.fields
      assert stats.sample == %{"a" => 1, "b" => 2}
    end

    test "handles empty dataset" do
      stats = DatasetBuilder.stats([])

      assert stats.count == 0
      assert stats.fields == []
      assert stats.sample == nil
    end
  end

  describe "from_hub/2" do
    test "returns error when hf_datasets_ex not available" do
      result = DatasetBuilder.from_hub("some/dataset")

      # Will error because HfDatasetsEx is not loaded in test context
      assert {:error, _} = result
    end
  end

  describe "chaining transformations" do
    test "can chain multiple transformations" do
      {:ok, dataset} =
        DatasetBuilder.from_list(
          Enum.map(1..100, fn i -> %{"value" => i, "category" => rem(i, 3)} end)
        )

      result =
        dataset
        |> DatasetBuilder.filter(fn row -> row["category"] == 0 end)
        |> DatasetBuilder.map(fn row -> Map.put(row, "doubled", row["value"] * 2) end)
        |> DatasetBuilder.take(5)

      assert length(result) == 5
      assert Enum.all?(result, fn row -> row["category"] == 0 end)
      assert Enum.all?(result, fn row -> row["doubled"] == row["value"] * 2 end)
    end
  end
end
