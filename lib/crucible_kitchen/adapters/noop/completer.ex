defmodule CrucibleKitchen.Adapters.Noop.Completer do
  @moduledoc """
  No-op adapter for the Completer port.

  Returns mock completions for testing and development.
  Useful when you don't have access to a real LLM API.

  ## Configuration

  ```elixir
  config :crucible_kitchen, :completer,
    adapter: CrucibleKitchen.Adapters.Noop.Completer,
    opts: [
      response_text: "Mock response",
      latency_ms: 100,
      fail_rate: 0.0
    ]
  ```
  """

  @behaviour CrucibleKitchen.Ports.Completer

  @impl true
  def complete(opts, messages, completion_opts) do
    maybe_simulate_latency(opts)

    if should_fail?(opts) do
      {:error, :simulated_failure}
    else
      result = build_result(opts, messages, completion_opts)
      {:ok, result}
    end
  end

  @impl true
  def complete_batch(opts, prompts, completion_opts) do
    maybe_simulate_latency(opts)

    if should_fail?(opts) do
      {:error, :simulated_failure}
    else
      results =
        Enum.map(prompts, fn messages ->
          build_result(opts, messages, completion_opts)
        end)

      {:ok, results}
    end
  end

  @impl true
  def stream_complete(opts, messages, _completion_opts) do
    maybe_simulate_latency(opts)

    if should_fail?(opts) do
      {:error, :simulated_failure}
    else
      response_text = get_response_text(opts, messages)
      # Split into chunks for streaming simulation
      chunks =
        response_text
        |> String.graphemes()
        |> Enum.chunk_every(5)
        |> Enum.map(&Enum.join/1)
        |> Enum.with_index()
        |> Enum.map(fn {text, idx} ->
          %{
            text: text,
            finish_reason: nil,
            index: idx
          }
        end)

      # Add final chunk with finish reason
      final_chunk = %{
        text: "",
        finish_reason: :stop,
        index: length(chunks)
      }

      {:ok, Stream.concat(chunks, [final_chunk])}
    end
  end

  @impl true
  def supports_streaming?(_opts), do: true

  @impl true
  def supports_tools?(_opts), do: true

  @impl true
  def get_model(opts), do: Keyword.get(opts, :model, "noop-mock-model")

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp build_result(opts, messages, completion_opts) do
    response_text = get_response_text(opts, messages)
    max_tokens = Keyword.get(completion_opts, :max_tokens, 1024)

    # Truncate if needed
    truncated =
      if String.length(response_text) > max_tokens * 4 do
        # Rough chars per token estimate
        String.slice(response_text, 0, max_tokens * 4)
      else
        response_text
      end

    finish_reason = if truncated == response_text, do: :stop, else: :length

    prompt_tokens = estimate_tokens(messages)
    completion_tokens = div(String.length(truncated), 4)

    %{
      text: truncated,
      finish_reason: finish_reason,
      usage: %{
        prompt_tokens: prompt_tokens,
        completion_tokens: completion_tokens,
        total_tokens: prompt_tokens + completion_tokens
      },
      logprobs: nil,
      tool_calls: nil
    }
  end

  defp get_response_text(opts, messages) do
    case Keyword.get(opts, :response_text) do
      nil -> generate_default_response(messages)
      text when is_binary(text) -> text
      fun when is_function(fun, 1) -> fun.(messages)
    end
  end

  defp generate_default_response(messages) do
    last_message = List.last(messages)

    content =
      case last_message do
        %{content: c} when is_binary(c) -> c
        %{"content" => c} when is_binary(c) -> c
        _ -> "unknown"
      end

    "This is a mock response to: #{String.slice(content, 0, 50)}..."
  end

  defp estimate_tokens(messages) do
    messages
    |> Enum.map(fn msg ->
      content =
        case msg do
          %{content: c} when is_binary(c) -> c
          %{"content" => c} when is_binary(c) -> c
          _ -> ""
        end

      # Rough estimate: 4 chars per token
      div(String.length(content), 4) + 4
    end)
    |> Enum.sum()
  end

  defp maybe_simulate_latency(opts) do
    case Keyword.get(opts, :latency_ms, 0) do
      0 -> :ok
      ms -> Process.sleep(ms)
    end
  end

  defp should_fail?(opts) do
    fail_rate = Keyword.get(opts, :fail_rate, 0.0)
    :rand.uniform() < fail_rate
  end
end
