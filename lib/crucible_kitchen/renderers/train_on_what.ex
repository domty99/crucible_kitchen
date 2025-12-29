defmodule CrucibleKitchen.Renderers.TrainOnWhat do
  @moduledoc """
  Enum controlling which tokens receive loss during supervised training.

  This determines the training signal by assigning weights (0.0 or 1.0)
  to different parts of the conversation.

  ## Options

  - `:last_assistant_message` - Only the final assistant response
  - `:all_assistant_messages` - All assistant responses in the conversation
  - `:all_messages` - All messages (user, assistant, system)
  - `:all_tokens` - Every token including headers and formatting
  - `:all_user_and_system_messages` - Only user and system messages
  - `:customized` - Per-message control via `trainable` field

  ## Examples

      # Standard SFT: train on final response only
      Renderers.build_supervised_example(renderer, messages,
        train_on_what: :last_assistant_message
      )

      # Multi-turn: train on all assistant responses
      Renderers.build_supervised_example(renderer, messages,
        train_on_what: :all_assistant_messages
      )

      # Customized: per-message control
      messages = [
        Message.user("Question", trainable: false),
        Message.assistant("Answer", trainable: true)
      ]
      Renderers.build_supervised_example(renderer, messages,
        train_on_what: :customized
      )
  """

  @type t ::
          :last_assistant_message
          | :all_assistant_messages
          | :all_messages
          | :all_tokens
          | :all_user_and_system_messages
          | :customized

  @valid_values [
    :last_assistant_message,
    :all_assistant_messages,
    :all_messages,
    :all_tokens,
    :all_user_and_system_messages,
    :customized
  ]

  @doc """
  Normalize a train_on_what value, accepting atoms or strings.
  """
  @spec normalize(t() | String.t()) :: t()
  def normalize(value) when value in @valid_values, do: value

  def normalize(value) when is_binary(value) do
    value
    |> String.downcase()
    |> String.to_existing_atom()
    |> normalize()
  rescue
    e in ArgumentError ->
      reraise ArgumentError,
              "Invalid train_on_what value: #{inspect(value)} (#{Exception.message(e)})",
              __STACKTRACE__
  end

  def normalize(value) do
    raise ArgumentError, "Invalid train_on_what value: #{inspect(value)}"
  end

  @doc """
  Check if a value is a valid train_on_what option.
  """
  @spec valid?(term()) :: boolean()
  def valid?(value) when value in @valid_values, do: true
  def valid?(_), do: false

  @doc """
  List all valid options.
  """
  @spec values() :: [t()]
  def values, do: @valid_values
end
