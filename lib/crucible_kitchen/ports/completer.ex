defmodule CrucibleKitchen.Ports.Completer do
  @moduledoc """
  Port for text completion/sampling from language models.

  This port defines the interface for generating text completions from
  language models, supporting both single completions and batched inference.

  ## Use Cases

  - Interactive chat completion
  - Batch inference for evaluation
  - Structured output generation (JSON, code)
  - Tool use / function calling

  ## Implementations

  Adapters should implement:
  - `complete/3` - Single completion
  - `complete_batch/3` - Batch completions
  - `stream_complete/3` - Streaming completion (optional)

  ## Examples

      # Single completion
      {:ok, result} = Completer.complete(adapter, messages, opts)

      # Batch completion for evaluation
      {:ok, results} = Completer.complete_batch(adapter, prompts, opts)

      # Streaming
      {:ok, stream} = Completer.stream_complete(adapter, messages, opts)
      Enum.each(stream, fn chunk -> IO.write(chunk.text) end)
  """

  alias CrucibleKitchen.Renderers.Message

  @type adapter :: module()
  @type adapter_opts :: keyword()

  @type completion_opts :: [
          temperature: float(),
          max_tokens: pos_integer(),
          top_p: float(),
          top_k: pos_integer(),
          stop_sequences: [String.t()],
          presence_penalty: float(),
          frequency_penalty: float(),
          logprobs: boolean(),
          n: pos_integer(),
          seed: integer() | nil,
          json_mode: boolean(),
          tools: [map()],
          tool_choice: String.t() | map()
        ]

  @type completion_result :: %{
          text: String.t(),
          finish_reason: :stop | :length | :tool_calls | :content_filter | nil,
          usage: %{
            prompt_tokens: non_neg_integer(),
            completion_tokens: non_neg_integer(),
            total_tokens: non_neg_integer()
          },
          logprobs: [%{token: String.t(), logprob: float()}] | nil,
          tool_calls: [map()] | nil
        }

  @type stream_chunk :: %{
          text: String.t(),
          finish_reason: :stop | :length | :tool_calls | :content_filter | nil,
          index: non_neg_integer()
        }

  @doc """
  Generate a completion for a list of messages.

  ## Parameters

  - `adapter` - The adapter module implementing this port
  - `messages` - List of Message structs or maps representing the conversation
  - `opts` - Completion options (see `t:completion_opts/0`)

  ## Returns

  - `{:ok, result}` - Successful completion
  - `{:error, reason}` - Completion failed
  """
  @callback complete(adapter_opts(), [Message.t() | map()], completion_opts()) ::
              {:ok, completion_result()} | {:error, term()}

  @doc """
  Generate completions for multiple prompts in batch.

  More efficient than calling complete/3 multiple times.

  ## Parameters

  - `adapter` - The adapter module
  - `prompts` - List of message lists (each is a conversation)
  - `opts` - Completion options

  ## Returns

  - `{:ok, results}` - List of completion results
  - `{:error, reason}` - Batch failed
  """
  @callback complete_batch(adapter_opts(), [[Message.t() | map()]], completion_opts()) ::
              {:ok, [completion_result()]} | {:error, term()}

  @doc """
  Generate a streaming completion.

  Returns a stream of chunks that can be consumed incrementally.

  ## Parameters

  - `adapter` - The adapter module
  - `messages` - Conversation messages
  - `opts` - Completion options

  ## Returns

  - `{:ok, stream}` - Enumerable stream of chunks
  - `{:error, reason}` - Stream creation failed
  """
  @callback stream_complete(adapter_opts(), [Message.t() | map()], completion_opts()) ::
              {:ok, Enumerable.t()} | {:error, term()}

  @doc """
  Check if the adapter supports streaming.
  """
  @callback supports_streaming?(adapter_opts()) :: boolean()

  @doc """
  Check if the adapter supports tool/function calling.
  """
  @callback supports_tools?(adapter_opts()) :: boolean()

  @doc """
  Get the model name/identifier being used.
  """
  @callback get_model(adapter_opts()) :: String.t()

  @optional_callbacks [stream_complete: 3, supports_streaming?: 1, supports_tools?: 1]

  # ==========================================================================
  # Convenience Functions
  # ==========================================================================

  @doc """
  Complete a single text prompt (convenience wrapper).

  Wraps the text in a user message and calls complete/3.
  """
  @spec complete_text(adapter(), adapter_opts(), String.t(), completion_opts()) ::
          {:ok, String.t()} | {:error, term()}
  def complete_text(adapter, opts, text, completion_opts \\ []) do
    messages = [Message.user(text)]

    case adapter.complete(opts, messages, completion_opts) do
      {:ok, result} -> {:ok, result.text}
      error -> error
    end
  end

  @doc """
  Complete with a system prompt and user message.
  """
  @spec complete_with_system(
          adapter(),
          adapter_opts(),
          String.t(),
          String.t(),
          completion_opts()
        ) ::
          {:ok, completion_result()} | {:error, term()}
  def complete_with_system(adapter, opts, system_prompt, user_message, completion_opts \\ []) do
    messages = [
      Message.system(system_prompt),
      Message.user(user_message)
    ]

    adapter.complete(opts, messages, completion_opts)
  end

  @doc """
  Create default completion options.
  """
  @spec default_opts() :: completion_opts()
  def default_opts do
    [
      temperature: 0.7,
      max_tokens: 1024,
      top_p: 1.0,
      stop_sequences: [],
      logprobs: false,
      n: 1
    ]
  end

  @doc """
  Merge user options with defaults.
  """
  @spec merge_opts(completion_opts()) :: completion_opts()
  def merge_opts(opts) do
    Keyword.merge(default_opts(), opts)
  end
end
