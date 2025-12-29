defmodule CrucibleKitchen.Renderers.RenderedMessage do
  @moduledoc """
  Container for a rendered message's token chunks.

  When a message is rendered to tokens, it's split into parts for masking:

  - `prefix` - Header with role (e.g., `<|start_header_id|>assistant<|end_header_id|>`)
  - `content` - The actual message content tokens
  - `suffix` - Optional end-of-turn marker (e.g., `<|eot_id|>`)

  This separation allows fine-grained control over which tokens receive
  loss during training.
  """

  alias CrucibleKitchen.Types.EncodedTextChunk

  @type chunk :: EncodedTextChunk.t() | nil

  @type t :: %__MODULE__{
          prefix: chunk(),
          content: [chunk()],
          suffix: chunk()
        }

  defstruct [:prefix, :suffix, content: []]

  @doc """
  Create a new rendered message.
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      prefix: Keyword.get(opts, :prefix),
      content: Keyword.get(opts, :content, []),
      suffix: Keyword.get(opts, :suffix)
    }
  end

  @doc """
  Get the total token count across all chunks.
  """
  @spec token_count(t()) :: non_neg_integer()
  def token_count(%__MODULE__{} = rendered) do
    prefix_count = if rendered.prefix, do: length(rendered.prefix.tokens), else: 0
    suffix_count = if rendered.suffix, do: length(rendered.suffix.tokens), else: 0

    content_count =
      rendered.content
      |> Enum.reject(&is_nil/1)
      |> Enum.map(&length(&1.tokens))
      |> Enum.sum()

    prefix_count + content_count + suffix_count
  end

  @doc """
  Get all tokens as a flat list.
  """
  @spec all_tokens(t()) :: [integer()]
  def all_tokens(%__MODULE__{} = rendered) do
    prefix_tokens = if rendered.prefix, do: rendered.prefix.tokens, else: []
    suffix_tokens = if rendered.suffix, do: rendered.suffix.tokens, else: []

    content_tokens =
      rendered.content
      |> Enum.reject(&is_nil/1)
      |> Enum.flat_map(& &1.tokens)

    prefix_tokens ++ content_tokens ++ suffix_tokens
  end
end
