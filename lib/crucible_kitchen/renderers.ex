defmodule CrucibleKitchen.Renderers do
  @moduledoc """
  Token <-> chat message conversion for different model families.

  Renderers handle the conversion between structured chat messages and
  token sequences for training and sampling. Each model family has its
  own chat template format.

  ## Available Renderers

  | Renderer | Model Family | Format |
  |----------|--------------|--------|
  | `RoleColon` | DeepSeek, Anthropic-style | `User: ... Assistant: ...` |
  | `Llama3` | Llama 3.x | `<\|start_header_id\|>...<\|eot_id\|>` |
  | `Qwen3` | Qwen 3.x | `<\|im_start\|>...<\|im_end\|>` with thinking |
  | `Mistral` | Mistral/Mixtral | `[INST]...[/INST]` |

  ## Usage

      alias CrucibleKitchen.Renderers
      alias CrucibleKitchen.Renderers.{Message, TrainOnWhat}

      # Create a renderer with a tokenizer
      {:ok, renderer} = Renderers.new(:llama3, tokenizer)

      # Build training example
      messages = [
        Message.system("You are helpful."),
        Message.user("Hello!"),
        Message.assistant("Hi there!")
      ]

      {:ok, model_input, weights} = Renderers.build_supervised_example(
        renderer,
        messages,
        train_on_what: :all_assistant_messages
      )

      # Build sampling prompt
      {:ok, model_input} = Renderers.build_generation_prompt(renderer, messages)

  ## TrainOnWhat Options

  Controls which tokens receive loss during training:

  - `:last_assistant_message` - Only the final assistant turn
  - `:all_assistant_messages` - All assistant turns
  - `:all_messages` - All turns including user/system
  - `:all_tokens` - Every token including headers
  - `:all_user_and_system_messages` - Only user and system turns
  - `:customized` - Per-message `trainable` field
  """

  alias CrucibleKitchen.Renderers.{Message, RenderedMessage, TrainOnWhat}
  alias CrucibleKitchen.Types.{EncodedTextChunk, ModelInput}

  @type tokenizer :: term()
  @type renderer :: %{
          type: atom(),
          tokenizer: tokenizer(),
          opts: keyword()
        }

  @doc """
  Create a new renderer for the given model family.

  ## Parameters

  - `type` - Renderer type (`:llama3`, `:qwen3`, `:role_colon`, `:mistral`)
  - `tokenizer` - Tokenizer handle from adapter-specific helper
  - `opts` - Renderer-specific options

  ## Options

  For `:qwen3`:
  - `:strip_thinking_from_history` - Strip `<think>` blocks from history (default: true)

  ## Examples

      {:ok, renderer} = Renderers.new(:llama3, tokenizer)
      {:ok, renderer} = Renderers.new(:qwen3, tokenizer, strip_thinking_from_history: false)
  """
  @spec new(atom(), tokenizer(), keyword()) :: {:ok, renderer()} | {:error, term()}
  def new(type, tokenizer, opts \\ [])
      when type in [:llama3, :qwen3, :qwen3_no_thinking, :role_colon, :mistral] do
    {:ok, %{type: type, tokenizer: tokenizer, opts: opts}}
  end

  @doc """
  Get the recommended renderer for a model name.
  """
  @spec recommended_for(String.t()) :: atom()
  def recommended_for(model_name) do
    model_lower = String.downcase(model_name)

    cond do
      String.contains?(model_lower, "llama-3") or String.contains?(model_lower, "llama3") ->
        :llama3

      String.contains?(model_lower, "qwen3") or String.contains?(model_lower, "qwen-3") ->
        :qwen3

      String.contains?(model_lower, "mistral") or String.contains?(model_lower, "mixtral") ->
        :mistral

      String.contains?(model_lower, "deepseek") ->
        :role_colon

      true ->
        :role_colon
    end
  end

  @doc """
  Build a generation prompt for sampling.

  Renders all messages and adds a partial assistant turn for the model to complete.
  """
  @spec build_generation_prompt(renderer(), [Message.t()], keyword()) ::
          {:ok, ModelInput.t()} | {:error, term()}
  def build_generation_prompt(renderer, messages, opts \\ []) do
    role = Keyword.get(opts, :role, "assistant")
    prefill = Keyword.get(opts, :prefill)

    chunks = bos_chunks(renderer)

    chunks =
      messages
      |> Enum.with_index()
      |> Enum.reduce(chunks, fn {message, idx}, acc ->
        rendered = render_message(renderer, idx, message, is_last: false)
        add_rendered_chunks(acc, rendered)
      end)

    # Add partial assistant turn
    partial = Message.new(role, "")
    rendered = render_message(renderer, length(messages), partial, is_last: false)
    chunks = add_prefix_only(chunks, rendered)

    # Add prefill if provided
    chunks =
      if prefill do
        tokens = encode(renderer.tokenizer, prefill, add_special_tokens: false)
        chunks ++ [%EncodedTextChunk{tokens: tokens}]
      else
        chunks
      end

    {:ok, %ModelInput{chunks: chunks}}
  end

  @doc """
  Build a supervised training example with weights.

  Returns the model input and a list of weights (0.0 or 1.0) for each token.
  """
  @spec build_supervised_example(renderer(), [Message.t()], keyword()) ::
          {:ok, ModelInput.t(), [float()]} | {:error, term()}
  def build_supervised_example(renderer, messages, opts \\ []) do
    train_on_what = Keyword.get(opts, :train_on_what, :last_assistant_message)
    train_on_what = TrainOnWhat.normalize(train_on_what)

    chunks_weights = bos_chunks_weighted(renderer)
    num_messages = length(messages)

    chunks_weights =
      messages
      |> Enum.with_index()
      |> Enum.reduce(chunks_weights, fn {message, idx}, acc ->
        is_last = idx == num_messages - 1
        rendered = render_message(renderer, idx, message, is_last: is_last)
        ob_weight = if train_on_what == :all_tokens, do: 1.0, else: 0.0
        action_weight = compute_action_weight(train_on_what, message, is_last)
        accumulate_rendered_weights(acc, rendered, ob_weight, action_weight, is_last)
      end)

    {chunks, weights} = unzip_chunks_weights(chunks_weights)
    {:ok, %ModelInput{chunks: chunks}, weights}
  end

  defp compute_action_weight(:last_assistant_message, %{role: "assistant"}, true), do: 1.0
  defp compute_action_weight(:all_assistant_messages, %{role: "assistant"}, _), do: 1.0
  defp compute_action_weight(:all_messages, _, _), do: 1.0
  defp compute_action_weight(:all_tokens, _, _), do: 1.0

  defp compute_action_weight(:all_user_and_system_messages, %{role: role}, _)
       when role in ["user", "system"],
       do: 1.0

  defp compute_action_weight(:customized, %{trainable: true}, _), do: 1.0
  defp compute_action_weight(_, _, _), do: 0.0

  defp accumulate_rendered_weights(acc, rendered, ob_weight, action_weight, is_last) do
    acc = add_chunk_weighted(acc, rendered.prefix, ob_weight)
    acc = Enum.reduce(rendered.content, acc, &add_chunk_weighted(&2, &1, action_weight))

    if is_last and rendered.suffix do
      add_chunk_weighted(acc, rendered.suffix, action_weight)
    else
      acc
    end
  end

  @doc """
  Render a single message to tokens.
  """
  @spec render_message(renderer(), non_neg_integer(), Message.t(), keyword()) ::
          RenderedMessage.t()
  def render_message(renderer, idx, message, opts \\ []) do
    is_last = Keyword.get(opts, :is_last, false)

    case renderer.type do
      :role_colon -> render_role_colon(renderer, idx, message, is_last)
      :llama3 -> render_llama3(renderer, idx, message, is_last)
      :qwen3 -> render_qwen3(renderer, idx, message, is_last)
      :qwen3_no_thinking -> render_qwen3_no_thinking(renderer, idx, message, is_last)
      :mistral -> render_mistral(renderer, idx, message, is_last)
    end
  end

  @doc """
  Get stop sequences for the renderer.
  """
  @spec get_stop_sequences(renderer()) :: [String.t()] | [integer()]
  def get_stop_sequences(renderer) do
    case renderer.type do
      :role_colon -> ["\n\nUser:"]
      :llama3 -> get_stop_token_ids(renderer, "<|eot_id|>")
      :qwen3 -> get_stop_token_ids(renderer, "<|im_end|>")
      :qwen3_no_thinking -> get_stop_token_ids(renderer, "<|im_end|>")
      :mistral -> ["</s>", "[/INST]"]
    end
  end

  @doc """
  Parse a response from token IDs back to a message.
  """
  @spec parse_response(renderer(), [integer()]) ::
          {:ok, Message.t(), boolean()} | {:error, term()}
  def parse_response(renderer, response) do
    case renderer.type do
      :role_colon -> parse_role_colon_response(renderer, response)
      :llama3 -> parse_stop_token_response(renderer, response, "<|eot_id|>")
      :qwen3 -> parse_qwen3_response(renderer, response)
      :qwen3_no_thinking -> parse_stop_token_response(renderer, response, "<|im_end|>")
      :mistral -> parse_mistral_response(renderer, response)
    end
  end

  # ==========================================================================
  # Private: Role Colon Renderer
  # ==========================================================================

  defp render_role_colon(renderer, _idx, message, _is_last) do
    role_str = String.capitalize(message.role)
    ob_str = "#{role_str}:"
    ac_str = " #{message.content}\n\n"

    prefix = encode_chunk(renderer.tokenizer, ob_str)
    content = [encode_chunk(renderer.tokenizer, ac_str)]
    suffix = encode_chunk(renderer.tokenizer, "User:")

    %RenderedMessage{prefix: prefix, content: content, suffix: suffix}
  end

  defp parse_role_colon_response(renderer, response) do
    str_response = decode(renderer.tokenizer, response)

    case String.split(str_response, "\n\nUser:", parts: 2) do
      [content] ->
        {:ok, Message.assistant(String.trim(content)), false}

      [content, _rest] ->
        {:ok, Message.assistant(String.trim(content)), true}
    end
  end

  # ==========================================================================
  # Private: Llama3 Renderer
  # ==========================================================================

  defp render_llama3(renderer, _idx, message, _is_last) do
    ob_str = "<|start_header_id|>#{message.role}<|end_header_id|>\n\n"
    ac_str = "#{message.content}<|eot_id|>"

    prefix = encode_chunk(renderer.tokenizer, ob_str)
    content = [encode_chunk(renderer.tokenizer, ac_str)]

    %RenderedMessage{prefix: prefix, content: content, suffix: nil}
  end

  # ==========================================================================
  # Private: Qwen3 Renderer
  # ==========================================================================

  defp render_qwen3(renderer, idx, message, is_last) do
    strip_thinking = Keyword.get(renderer.opts, :strip_thinking_from_history, true)
    maybe_newline = if idx > 0, do: "\n", else: ""
    role = if message.role == "tool", do: "user", else: message.role

    ac_content =
      message.content
      |> maybe_wrap_tool_response(message.role)
      |> maybe_strip_thinking(strip_thinking, message.role, is_last)
      |> maybe_append_tool_calls(message.tool_calls)

    ob_str = build_qwen3_prefix(maybe_newline, role, ac_content, message.role, is_last)
    ac_content = ac_content <> "<|im_end|>"

    prefix = encode_chunk(renderer.tokenizer, ob_str)
    content = [encode_chunk(renderer.tokenizer, ac_content)]

    %RenderedMessage{prefix: prefix, content: content, suffix: nil}
  end

  defp maybe_wrap_tool_response(content, "tool"),
    do: "<tool_response>\n#{content}\n</tool_response>"

  defp maybe_wrap_tool_response(content, _role), do: content

  defp maybe_strip_thinking(content, true, "assistant", false) do
    if String.contains?(content, "</think>") do
      content |> String.split("</think>", parts: 2) |> List.last() |> String.trim_leading()
    else
      content
    end
  end

  defp maybe_strip_thinking(content, _strip, _role, _is_last), do: content

  defp maybe_append_tool_calls(content, nil), do: content
  defp maybe_append_tool_calls(content, []), do: content

  defp maybe_append_tool_calls(content, tool_calls) do
    tool_call_strs =
      Enum.map(tool_calls, fn tc ->
        payload = Jason.encode!(%{name: tc.name, arguments: tc.arguments})
        "<tool_call>\n#{payload}\n</tool_call>"
      end)

    content <> "\n" <> Enum.join(tool_call_strs, "\n")
  end

  defp build_qwen3_prefix(maybe_newline, role, content, "assistant", true) do
    base = "#{maybe_newline}<|im_start|>#{role}\n"
    if String.contains?(content, "<think>"), do: base, else: base <> "<think>\n"
  end

  defp build_qwen3_prefix(maybe_newline, role, _content, _msg_role, _is_last) do
    "#{maybe_newline}<|im_start|>#{role}\n"
  end

  defp render_qwen3_no_thinking(renderer, idx, message, is_last) do
    # Add empty thinking block if not present
    message =
      if message.role == "assistant" and not String.contains?(message.content, "<think>") do
        %{message | content: "<think>\n\n</think>\n\n" <> message.content}
      else
        message
      end

    render_qwen3(renderer, idx, message, is_last)
  end

  defp parse_qwen3_response(renderer, response) do
    case parse_stop_token_response(renderer, response, "<|im_end|>") do
      {:ok, message, false} ->
        {:ok, message, false}

      {:ok, message, true} ->
        maybe_extract_tool_calls(message)
    end
  end

  defp maybe_extract_tool_calls(message) do
    tool_calls = extract_tool_calls(message.content)

    if tool_calls == [] do
      {:ok, message, true}
    else
      clean_content = Regex.replace(~r/\n?<tool_call>.*?<\/tool_call>/s, message.content, "")
      {:ok, %{message | content: String.trim(clean_content), tool_calls: tool_calls}, true}
    end
  end

  defp extract_tool_calls(content) do
    ~r/<tool_call>(.*?)<\/tool_call>/s
    |> Regex.scan(content)
    |> Enum.map(&parse_tool_call/1)
    |> Enum.reject(&is_nil/1)
  end

  defp parse_tool_call([_, tc_str]) do
    case Jason.decode(String.trim(tc_str)) do
      {:ok, %{"name" => name, "arguments" => args}} -> %{name: name, arguments: args, id: nil}
      _ -> nil
    end
  end

  # ==========================================================================
  # Private: Mistral Renderer
  # ==========================================================================

  defp render_mistral(renderer, idx, message, _is_last) do
    case message.role do
      "user" ->
        maybe_space = if idx > 0, do: " ", else: ""
        ob_str = "#{maybe_space}[INST] "
        ac_str = "#{message.content} [/INST]"

        prefix = encode_chunk(renderer.tokenizer, ob_str)
        content = [encode_chunk(renderer.tokenizer, ac_str)]
        %RenderedMessage{prefix: prefix, content: content, suffix: nil}

      "assistant" ->
        content = [encode_chunk(renderer.tokenizer, message.content <> "</s>")]
        %RenderedMessage{prefix: nil, content: content, suffix: nil}

      "system" ->
        # System message is wrapped in first user message
        content = [encode_chunk(renderer.tokenizer, "<<SYS>>\n#{message.content}\n<</SYS>>\n\n")]
        %RenderedMessage{prefix: nil, content: content, suffix: nil}

      _ ->
        content = [encode_chunk(renderer.tokenizer, message.content)]
        %RenderedMessage{prefix: nil, content: content, suffix: nil}
    end
  end

  defp parse_mistral_response(renderer, response) do
    str_response = decode(renderer.tokenizer, response)

    cond do
      String.contains?(str_response, "</s>") ->
        [content | _] = String.split(str_response, "</s>", parts: 2)
        {:ok, Message.assistant(String.trim(content)), true}

      String.contains?(str_response, "[/INST]") ->
        [content | _] = String.split(str_response, "[/INST]", parts: 2)
        {:ok, Message.assistant(String.trim(content)), true}

      true ->
        {:ok, Message.assistant(String.trim(str_response)), false}
    end
  end

  # ==========================================================================
  # Private: Helpers
  # ==========================================================================

  defp bos_chunks(renderer) do
    case renderer.type do
      :role_colon ->
        case get_bos_token(renderer.tokenizer) do
          nil ->
            []

          bos ->
            [
              %EncodedTextChunk{
                tokens: encode(renderer.tokenizer, bos, add_special_tokens: false)
              }
            ]
        end

      :llama3 ->
        [
          %EncodedTextChunk{
            tokens: encode(renderer.tokenizer, "<|begin_of_text|>", add_special_tokens: false)
          }
        ]

      _ ->
        []
    end
  end

  defp bos_chunks_weighted(renderer) do
    bos_chunks(renderer)
    |> Enum.map(&{&1, 0.0})
  end

  defp encode_chunk(tokenizer, text) do
    tokens = encode(tokenizer, text, add_special_tokens: false)
    %EncodedTextChunk{tokens: tokens}
  end

  defp add_rendered_chunks(acc, rendered) do
    acc = if rendered.prefix, do: acc ++ [rendered.prefix], else: acc
    acc = acc ++ Enum.reject(rendered.content, &is_nil/1)
    if rendered.suffix, do: acc ++ [rendered.suffix], else: acc
  end

  defp add_prefix_only(acc, rendered) do
    if rendered.prefix, do: acc ++ [rendered.prefix], else: acc
  end

  defp add_chunk_weighted(acc, nil, _weight), do: acc

  defp add_chunk_weighted(acc, chunk, weight) do
    acc ++ [{chunk, weight}]
  end

  defp unzip_chunks_weights(chunks_weights) do
    {chunks, weight_lists} =
      Enum.reduce(chunks_weights, {[], []}, fn {chunk, weight}, {cs, ws} ->
        token_count = length(chunk.tokens)
        weights = List.duplicate(weight, token_count)
        {cs ++ [chunk], ws ++ weights}
      end)

    {chunks, weight_lists}
  end

  defp get_stop_token_ids(renderer, token_str) do
    [encode(renderer.tokenizer, token_str, add_special_tokens: false)]
  end

  defp parse_stop_token_response(renderer, response, stop_token_str) do
    stop_token =
      renderer.tokenizer
      |> encode(stop_token_str, add_special_tokens: false)
      |> List.first()

    stop_count = Enum.count(response, &(&1 == stop_token))

    cond do
      stop_count == 0 ->
        str_response = decode(renderer.tokenizer, response)
        {:ok, Message.assistant(str_response), false}

      stop_count == 1 ->
        idx = Enum.find_index(response, &(&1 == stop_token))
        str_response = decode(renderer.tokenizer, Enum.take(response, idx))
        {:ok, Message.assistant(str_response), true}

      true ->
        {:error, "Multiple stop tokens found in response (#{stop_count})"}
    end
  end

  # Tokenizer interface helpers
  defp encode(tokenizer, text, opts) do
    # This will be implemented based on the actual tokenizer adapter
    # For now, assume tokenizer has encode/2 function
    add_special = Keyword.get(opts, :add_special_tokens, true)

    cond do
      is_map(tokenizer) and Map.has_key?(tokenizer, :encode) ->
        tokenizer.encode.(text, add_special)

      is_function(tokenizer, 2) ->
        tokenizer.(text, add_special)

      true ->
        # Fallback for testing - simple char encoding
        text
        |> String.to_charlist()
        |> Enum.map(&rem(&1, 32_000))
    end
  end

  defp decode(tokenizer, tokens) do
    cond do
      is_map(tokenizer) and Map.has_key?(tokenizer, :decode) ->
        tokenizer.decode.(tokens)

      is_function(tokenizer, 1) ->
        tokenizer.(tokens)

      true ->
        # Fallback for testing
        Enum.map_join(tokens, &<<rem(&1 + 32, 95) + 32>>)
    end
  end

  defp get_bos_token(tokenizer) do
    if is_map(tokenizer) and Map.has_key?(tokenizer, :bos_token) do
      tokenizer.bos_token
    else
      nil
    end
  end
end
