defmodule CrucibleKitchen.Stages.InitTokenizer do
  @moduledoc """
  Stage for initializing the tokenizer from the training session.

  Gets the tokenizer from the training client and stores it for use
  in rendering messages to tokens.

  ## State Requirements

  - `:session` - Training session (from InitSession)

  ## State Updates

  - `:tokenizer` - Tokenizer handle for encoding/decoding
  """

  use CrucibleKitchen.Stage

  require Logger

  @impl true
  def name, do: :init_tokenizer

  @impl true
  def execute(context) do
    session = get_state(context, :session)

    case fetch_tokenizer(context, session) do
      {:ok, tokenizer} ->
        Logger.debug("[InitTokenizer] Tokenizer initialized")
        context = put_state(context, :tokenizer, tokenizer)
        {:ok, context}

      {:error, :not_implemented} ->
        # Some backends don't expose tokenizer directly
        Logger.debug("[InitTokenizer] Tokenizer not available from backend")
        {:ok, context}

      {:error, reason} ->
        Logger.error("[InitTokenizer] Failed: #{inspect(reason)}")
        {:error, {:tokenizer_init_failed, reason}}
    end
  end

  defp fetch_tokenizer(context, session) do
    case get_adapter(context, :training_client) do
      {module, opts} ->
        if function_exported?(module, :get_tokenizer, 2) do
          module.get_tokenizer(opts, session)
        else
          {:error, :not_implemented}
        end

      nil ->
        {:error, :missing_training_client}
    end
  end
end
