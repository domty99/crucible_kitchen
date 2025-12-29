defmodule CrucibleKitchen.Stages.Cleanup do
  @moduledoc """
  Stage for cleaning up resources at the end of training.

  Closes the training session and releases any allocated resources.

  ## State Requirements

  - `:session` - Training session to close
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :cleanup

  @impl true
  def execute(context) do
    session = get_state(context, :session)

    if session do
      Logger.info("[Cleanup] Closing training session")
      ports = get_train_ports(context)
      TrainingClient.close_session(ports, session)
    end

    context =
      context
      |> put_state(:session, nil)
      |> record_metric(:session_closed, 1)

    {:ok, context}
  end
end
