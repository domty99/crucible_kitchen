defmodule CrucibleKitchen.Adapters.Noop.ModelRegistryTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.ModelRegistry

  describe "register/2" do
    test "registers a model and returns it with an ID" do
      artifact = %{
        name: "test-model",
        version: "v1.0.0",
        artifact_uri: "s3://bucket/model.tar.gz",
        metadata: %{training_config: %{epochs: 3}},
        lineage: %{dataset: "test-dataset"}
      }

      assert {:ok, model} = ModelRegistry.register([], artifact)
      assert is_binary(model.id)
      assert model.name == "test-model"
      assert model.version == "v1.0.0"
      assert model.artifact_uri == "s3://bucket/model.tar.gz"
      assert model.metadata == artifact.metadata
      assert model.lineage == artifact.lineage
      assert %DateTime{} = model.created_at
    end

    test "generates unique IDs for each registration" do
      artifact1 = %{name: "model1", version: "v1"}
      artifact2 = %{name: "model2", version: "v2"}

      {:ok, model1} = ModelRegistry.register([], artifact1)
      {:ok, model2} = ModelRegistry.register([], artifact2)

      assert model1.id != model2.id
    end
  end

  describe "get/2" do
    test "retrieves a previously registered model" do
      artifact = %{name: "get-test-model", version: "v1.0.0"}
      {:ok, registered} = ModelRegistry.register([], artifact)

      assert {:ok, retrieved} = ModelRegistry.get([], registered.id)
      assert retrieved.id == registered.id
      assert retrieved.name == "get-test-model"
    end

    test "returns error for non-existent model" do
      assert {:error, :not_found} = ModelRegistry.get([], "non-existent-id")
    end
  end

  describe "list/1" do
    test "returns all registered models" do
      # Register some models
      {:ok, _} = ModelRegistry.register([], %{name: "list-model-1", version: "v1"})
      {:ok, _} = ModelRegistry.register([], %{name: "list-model-2", version: "v1"})

      assert {:ok, models} = ModelRegistry.list([])
      assert is_list(models)
    end

    test "filters by name" do
      {:ok, _} = ModelRegistry.register([], %{name: "filter-test", version: "v1"})
      {:ok, _} = ModelRegistry.register([], %{name: "other-model", version: "v1"})

      assert {:ok, filtered} = ModelRegistry.list(name: "filter-test")
      assert Enum.all?(filtered, fn m -> m.name == "filter-test" end)
    end

    test "limits results" do
      {:ok, _} = ModelRegistry.register([], %{name: "limit-test-1", version: "v1"})
      {:ok, _} = ModelRegistry.register([], %{name: "limit-test-2", version: "v1"})
      {:ok, _} = ModelRegistry.register([], %{name: "limit-test-3", version: "v1"})

      assert {:ok, limited} = ModelRegistry.list(limit: 2)
      assert length(limited) <= 2
    end
  end
end
