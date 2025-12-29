defmodule CrucibleKitchen.Stages.AwaitFuture do
  @moduledoc """
  Stage for awaiting an async operation future.

  Waits for a future stored in state and optionally stores the result.

  ## Options

  - `:key` - State key for the future to await (default: :future)
  - `:result_key` - State key to store the result (default: nil, discards result)

  ## Example

      stage(:await_fb, AwaitFuture, key: :fb_future, result_key: :fb_result)
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.TrainingClient

  require Logger

  @impl true
  def name, do: :await_future

  @impl true
  def execute(context) do
    # Get the stage options from context (passed via workflow)
    opts = get_state(context, :stage_opts, [])
    key = Keyword.get(opts, :key, :future)
    result_key = Keyword.get(opts, :result_key)

    future = get_state(context, key)

    if is_nil(future) do
      # No future to await, continue
      {:ok, context}
    else
      ports = get_train_ports(context)

      case TrainingClient.await(ports, future) do
        {:ok, result} ->
          context =
            context
            |> put_state(key, nil)
            |> maybe_store_result(result_key, result)

          {:ok, context}

        {:error, reason} ->
          Logger.error("[AwaitFuture] Await failed for #{key}: #{inspect(reason)}")
          {:error, {:await_failed, key, reason}}
      end
    end
  end

  defp maybe_store_result(context, nil, _result), do: context
  defp maybe_store_result(context, key, result), do: put_state(context, key, result)
end
