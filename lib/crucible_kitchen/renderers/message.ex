defmodule CrucibleKitchen.Renderers.Message do
  @moduledoc """
  Structured representation of a chat message.

  Messages are the building blocks of conversations for training and inference.
  Each message has a role and content, with optional tool calls and metadata.

  ## Roles

  Standard roles:
  - `"system"` - System instructions/context
  - `"user"` - User input
  - `"assistant"` - Model response
  - `"tool"` - Tool/function response

  ## Examples

      # Simple messages
      Message.system("You are a helpful assistant.")
      Message.user("What is 2+2?")
      Message.assistant("The answer is 4.")

      # With tool calls
      Message.assistant("Let me search for that.", tool_calls: [
        %{name: "search", arguments: %{query: "weather"}, id: "call_123"}
      ])

      # Tool response
      Message.tool("Sunny, 72°F", tool_call_id: "call_123")

      # Customized training
      Message.user("Hello", trainable: true)
  """

  @type content_part ::
          %{type: :text, text: String.t()}
          | %{type: :image, image: String.t() | binary()}

  @type tool_call :: %{
          name: String.t(),
          arguments: map() | String.t(),
          id: String.t() | nil
        }

  @type t :: %__MODULE__{
          role: String.t(),
          content: String.t() | [content_part()],
          tool_calls: [tool_call()] | nil,
          tool_call_id: String.t() | nil,
          thinking: String.t() | nil,
          trainable: boolean() | nil,
          name: String.t() | nil
        }

  defstruct [
    :role,
    :content,
    :tool_calls,
    :tool_call_id,
    :thinking,
    :trainable,
    :name
  ]

  @doc """
  Create a new message with the given role and content.
  """
  @spec new(String.t(), String.t() | [content_part()], keyword()) :: t()
  def new(role, content, opts \\ []) do
    %__MODULE__{
      role: role,
      content: content,
      tool_calls: Keyword.get(opts, :tool_calls),
      tool_call_id: Keyword.get(opts, :tool_call_id),
      thinking: Keyword.get(opts, :thinking),
      trainable: Keyword.get(opts, :trainable),
      name: Keyword.get(opts, :name)
    }
  end

  @doc """
  Create a system message.
  """
  @spec system(String.t(), keyword()) :: t()
  def system(content, opts \\ []), do: new("system", content, opts)

  @doc """
  Create a user message.
  """
  @spec user(String.t() | [content_part()], keyword()) :: t()
  def user(content, opts \\ []), do: new("user", content, opts)

  @doc """
  Create an assistant message.
  """
  @spec assistant(String.t(), keyword()) :: t()
  def assistant(content, opts \\ []), do: new("assistant", content, opts)

  @doc """
  Create a tool response message.
  """
  @spec tool(String.t(), keyword()) :: t()
  def tool(content, opts \\ []), do: new("tool", content, opts)

  @doc """
  Check if the message content is text-only (not multimodal).
  """
  @spec text_only?(t()) :: boolean()
  def text_only?(%__MODULE__{content: content}) when is_binary(content), do: true

  def text_only?(%__MODULE__{content: [%{type: :text}]}), do: true
  def text_only?(_), do: false

  @doc """
  Extract text content, raising if multimodal.
  """
  @spec ensure_text!(t()) :: String.t()
  def ensure_text!(%__MODULE__{content: content}) when is_binary(content), do: content

  def ensure_text!(%__MODULE__{content: [%{type: :text, text: text}]}), do: text

  def ensure_text!(%__MODULE__{content: content}) when is_list(content) do
    raise ArgumentError,
          "Expected text content, got multimodal content with #{length(content)} parts"
  end

  @doc """
  Convert a map or keyword list to a Message struct.
  """
  @spec from_map(map() | keyword()) :: t()
  def from_map(data) when is_list(data) do
    from_map(Map.new(data))
  end

  def from_map(%{} = data) do
    %__MODULE__{
      role: data[:role] || data["role"],
      content: data[:content] || data["content"],
      tool_calls: data[:tool_calls] || data["tool_calls"],
      tool_call_id: data[:tool_call_id] || data["tool_call_id"],
      thinking: data[:thinking] || data["thinking"],
      trainable: data[:trainable] || data["trainable"],
      name: data[:name] || data["name"]
    }
  end

  @doc """
  Convert a Message to a plain map.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = message) do
    %{
      role: message.role,
      content: message.content
    }
    |> maybe_put(:tool_calls, message.tool_calls)
    |> maybe_put(:tool_call_id, message.tool_call_id)
    |> maybe_put(:thinking, message.thinking)
    |> maybe_put(:trainable, message.trainable)
    |> maybe_put(:name, message.name)
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)
end
