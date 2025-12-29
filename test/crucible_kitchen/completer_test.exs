defmodule CrucibleKitchen.CompleterTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Adapters.Noop.Completer, as: NoopCompleter
  alias CrucibleKitchen.Ports.Completer
  alias CrucibleKitchen.Renderers.Message

  describe "Completer port" do
    test "default_opts returns expected defaults" do
      opts = Completer.default_opts()

      assert opts[:temperature] == 0.7
      assert opts[:max_tokens] == 1024
      assert opts[:top_p] == 1.0
      assert opts[:stop_sequences] == []
      assert opts[:logprobs] == false
      assert opts[:n] == 1
    end

    test "merge_opts combines with defaults" do
      merged = Completer.merge_opts(temperature: 0.5, max_tokens: 512)

      assert merged[:temperature] == 0.5
      assert merged[:max_tokens] == 512
      assert merged[:top_p] == 1.0
    end
  end

  describe "Noop.Completer" do
    test "complete/3 returns mock result" do
      messages = [Message.user("Hello")]

      {:ok, result} = NoopCompleter.complete([], messages, [])

      assert is_binary(result.text)
      assert result.finish_reason == :stop
      assert is_map(result.usage)
      assert result.usage.prompt_tokens > 0
      assert result.usage.completion_tokens > 0
    end

    test "complete/3 with custom response_text" do
      opts = [response_text: "Custom response"]
      messages = [Message.user("Test")]

      {:ok, result} = NoopCompleter.complete(opts, messages, [])

      assert result.text == "Custom response"
    end

    test "complete/3 with function response_text" do
      opts = [
        response_text: fn messages ->
          last = List.last(messages)
          "Echo: #{last.content}"
        end
      ]

      messages = [Message.user("Hello world")]

      {:ok, result} = NoopCompleter.complete(opts, messages, [])

      assert result.text == "Echo: Hello world"
    end

    test "complete/3 truncates long responses" do
      opts = [response_text: String.duplicate("a", 10_000)]
      messages = [Message.user("Test")]

      {:ok, result} = NoopCompleter.complete(opts, messages, max_tokens: 100)

      assert result.finish_reason == :length
      # 100 tokens * 4 chars = 400 chars
      assert String.length(result.text) <= 400
    end

    test "complete_batch/3 returns results for all prompts" do
      prompts = [
        [Message.user("First")],
        [Message.user("Second")],
        [Message.user("Third")]
      ]

      {:ok, results} = NoopCompleter.complete_batch([], prompts, [])

      assert length(results) == 3
      assert Enum.all?(results, &is_binary(&1.text))
    end

    test "stream_complete/3 returns stream of chunks" do
      opts = [response_text: "Hello world!"]
      messages = [Message.user("Test")]

      {:ok, stream} = NoopCompleter.stream_complete(opts, messages, [])

      chunks = Enum.to_list(stream)

      # Should have content chunks plus final
      assert length(chunks) >= 2

      # Last chunk should have finish_reason
      last = List.last(chunks)
      assert last.finish_reason == :stop

      # Concatenated text should equal original
      text = Enum.map_join(chunks, & &1.text)
      assert text == "Hello world!"
    end

    test "supports_streaming?/1 returns true" do
      assert NoopCompleter.supports_streaming?([])
    end

    test "supports_tools?/1 returns true" do
      assert NoopCompleter.supports_tools?([])
    end

    test "get_model/1 returns default model name" do
      assert NoopCompleter.get_model([]) == "noop-mock-model"
    end

    test "get_model/1 returns custom model name" do
      assert NoopCompleter.get_model(model: "custom-model") == "custom-model"
    end

    test "complete/3 with fail_rate returns errors" do
      opts = [fail_rate: 1.0]
      messages = [Message.user("Test")]

      result = NoopCompleter.complete(opts, messages, [])

      assert result == {:error, :simulated_failure}
    end

    test "complete/3 works with map messages" do
      messages = [%{role: "user", content: "Hello"}]

      {:ok, result} = NoopCompleter.complete([], messages, [])

      assert is_binary(result.text)
    end

    test "complete/3 works with string key map messages" do
      messages = [%{"role" => "user", "content" => "Hello"}]

      {:ok, result} = NoopCompleter.complete([], messages, [])

      assert is_binary(result.text)
    end
  end

  describe "convenience functions" do
    test "complete_text/4 returns just the text" do
      {:ok, text} = Completer.complete_text(NoopCompleter, [], "Hello")

      assert is_binary(text)
    end

    test "complete_with_system/5 includes system message" do
      {:ok, result} =
        Completer.complete_with_system(
          NoopCompleter,
          [],
          "You are helpful.",
          "What is 2+2?"
        )

      assert is_binary(result.text)
    end
  end
end
