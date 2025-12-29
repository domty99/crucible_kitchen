defmodule CrucibleKitchen.Stages.SaveCheckpoint do
  @moduledoc """
  Stage for saving a training checkpoint.

  ## State Requirements

  - `:session` - Training session
  - `:global_step` - Current global step (used for checkpoint name)

  ## Configuration

  - `:checkpoint_prefix` - Prefix for checkpoint names (default: "step")
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :save_checkpoint

  @impl true
  def execute(context) do
    session = get_state(context, :session)
    global_step = get_state(context, :global_step, 0)
    prefix = get_config(context, :checkpoint_prefix, "step")

    checkpoint_name = "#{prefix}_#{String.pad_leading(Integer.to_string(global_step), 6, "0")}"

    Logger.info("[SaveCheckpoint] Saving checkpoint: #{checkpoint_name}")

    ports = get_train_ports(context)

    case TrainingClient.save_checkpoint(ports, session, checkpoint_name) do
      :ok ->
        context =
          context
          |> put_state(:last_checkpoint, checkpoint_name)
          |> record_metric(:checkpoint_saved, 1)

        {:ok, context}

      {:error, reason} ->
        Logger.warning("[SaveCheckpoint] Failed to save: #{inspect(reason)}")
        # Non-fatal - continue training
        {:ok, context}
    end
  end
end
