defmodule CrucibleKitchen.Recipes.SupervisedFineTuning do
  @moduledoc """
  Recipe for supervised fine-tuning (SFT).

  This is the standard recipe for instruction-following fine-tuning.
  It trains a model on input-output pairs using next-token prediction.

  ## When to Use

  - Fine-tuning a base model on instruction data
  - Adapting a model to a specific domain
  - Creating a task-specific assistant

  ## Training Flow

  1. Load and preprocess dataset
  2. Initialize training session with model
  3. For each epoch:
     - Train on batches
     - Evaluate on validation set
     - Save checkpoint if improved
  4. Finalize and return best model

  ## Example

      adapters = %{
        training_client: {TinkexCookbook.Adapters.TrainingClient, api_key: "..."},
        dataset_store: MyApp.Adapters.LocalDataset
      }

      {:ok, result} = CrucibleKitchen.run(
        CrucibleKitchen.Recipes.SupervisedFineTuning,
        %{
          model: "meta-llama/Llama-2-7b",
          dataset: "my_instructions",
          epochs: 3,
          batch_size: 8
        },
        adapters: adapters
      )
  """

  use CrucibleKitchen.Recipe

  @impl true
  def name, do: :supervised_finetuning

  @impl true
  def description do
    "Supervised fine-tuning for instruction-following models using next-token prediction"
  end

  @impl true
  def default_config do
    %{
      # Required - model identifier
      model: nil,
      # Required - dataset identifier
      dataset: nil,
      # Training params
      epochs: 1,
      batch_size: 4,
      learning_rate: 2.0e-5,
      warmup_steps: 100,
      max_steps: nil,
      gradient_accumulation_steps: 1,
      # LoRA params
      lora_rank: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      # Evaluation
      eval_split: "validation",
      eval_every_n_steps: 100,
      # Checkpointing
      checkpoint_every_n_steps: 500,
      save_best_only: true
    }
  end

  @impl true
  def required_adapters do
    [:training_client, :dataset_store]
  end

  @impl true
  def optional_adapters do
    [:metrics_store, :blob_store, :hub_client]
  end

  @impl true
  def workflow do
    # Returns a list of workflow steps conforming to Workflow.t()
    [
      {:stage, :load_dataset, CrucibleKitchen.Stages.Noop, []},
      {:stage, :init_session, CrucibleKitchen.Stages.Noop, []},
      {:loop, :training, [over: :epochs_range],
       [
         {:stage, :train_epoch, CrucibleKitchen.Stages.Noop, []},
         {:stage, :eval_epoch, CrucibleKitchen.Stages.Noop, []},
         {:stage, :checkpoint, CrucibleKitchen.Stages.Noop, []}
       ]},
      {:stage, :finalize, CrucibleKitchen.Stages.Noop, []}
    ]
  end

  def epochs_range(context) do
    epochs = CrucibleKitchen.Context.get_config(context, :epochs, 1)
    0..(epochs - 1)
  end

  @impl true
  def validate_config(config) do
    cond do
      is_nil(config[:model]) ->
        {:error, ":model is required - specify the model to fine-tune"}

      is_nil(config[:dataset]) ->
        {:error, ":dataset is required - specify the training dataset"}

      config[:epochs] < 1 ->
        {:error, ":epochs must be >= 1"}

      config[:batch_size] < 1 ->
        {:error, ":batch_size must be >= 1"}

      config[:learning_rate] <= 0 ->
        {:error, ":learning_rate must be positive"}

      config[:lora_rank] != nil and config[:lora_rank] < 1 ->
        {:error, ":lora_rank must be >= 1 when specified"}

      true ->
        :ok
    end
  end
end
