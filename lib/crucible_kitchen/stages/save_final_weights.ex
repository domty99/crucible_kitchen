defmodule CrucibleKitchen.Stages.SaveFinalWeights do
  @moduledoc """
  Stage for saving the final trained weights.

  Saves the model weights at the end of training.

  ## State Requirements

  - `:session` - Training session

  ## Configuration

  - `:final_weights_name` - Name for the final weights (default: "final_weights")
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :save_final_weights

  @impl true
  def execute(context) do
    session = get_state(context, :session)
    weights_name = get_config(context, :final_weights_name, "final_weights")

    Logger.info("[SaveFinalWeights] Saving final weights as: #{weights_name}")

    # Try save_checkpoint first, which most backends support
    ports = get_train_ports(context)

    case TrainingClient.save_checkpoint(ports, session, weights_name) do
      :ok ->
        Logger.info("[SaveFinalWeights] Saved weights: #{weights_name}")

        context =
          context
          |> put_state(:final_weights_path, weights_name)
          |> record_metric(:final_weights_saved, 1)

        {:ok, context}

      {:error, reason} ->
        Logger.warning("[SaveFinalWeights] Save failed: #{inspect(reason)}")
        # Non-fatal at end of training
        {:ok, context}
    end
  end
end
