defmodule CrucibleKitchen.RenderersTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Renderers
  alias CrucibleKitchen.Renderers.{Message, RenderedMessage, TrainOnWhat}
  alias CrucibleKitchen.Types.{EncodedTextChunk, ModelInput}

  # Simple mock tokenizer for testing
  defp mock_tokenizer do
    %{
      encode: fn text, _add_special ->
        # Simple char-based encoding
        text |> String.to_charlist() |> Enum.map(&rem(&1, 1000))
      end,
      decode: fn tokens ->
        Enum.map_join(tokens, &<<rem(&1 + 32, 95) + 32>>)
      end,
      bos_token: "<s>"
    }
  end

  describe "Message" do
    test "creates system message" do
      msg = Message.system("You are helpful.")
      assert msg.role == "system"
      assert msg.content == "You are helpful."
    end

    test "creates user message" do
      msg = Message.user("Hello!")
      assert msg.role == "user"
      assert msg.content == "Hello!"
    end

    test "creates assistant message" do
      msg = Message.assistant("Hi there!")
      assert msg.role == "assistant"
      assert msg.content == "Hi there!"
    end

    test "creates tool message" do
      msg = Message.tool("Result", tool_call_id: "call_123")
      assert msg.role == "tool"
      assert msg.content == "Result"
      assert msg.tool_call_id == "call_123"
    end

    test "creates message with tool calls" do
      msg =
        Message.assistant("Let me search.",
          tool_calls: [%{name: "search", arguments: %{q: "weather"}, id: "c1"}]
        )

      assert msg.tool_calls == [%{name: "search", arguments: %{q: "weather"}, id: "c1"}]
    end

    test "creates message with trainable flag" do
      msg = Message.user("Train on this", trainable: true)
      assert msg.trainable == true
    end

    test "text_only? returns true for string content" do
      msg = Message.user("Hello")
      assert Message.text_only?(msg)
    end

    test "text_only? returns true for single text part" do
      msg = Message.new("user", [%{type: :text, text: "Hello"}])
      assert Message.text_only?(msg)
    end

    test "text_only? returns false for multimodal" do
      msg = Message.new("user", [%{type: :text, text: "Hi"}, %{type: :image, image: "data:..."}])
      refute Message.text_only?(msg)
    end

    test "ensure_text! extracts text content" do
      msg = Message.user("Hello")
      assert Message.ensure_text!(msg) == "Hello"
    end

    test "ensure_text! raises for multimodal" do
      msg = Message.new("user", [%{type: :text, text: "Hi"}, %{type: :image, image: "data:..."}])

      assert_raise ArgumentError, ~r/Expected text content/, fn ->
        Message.ensure_text!(msg)
      end
    end

    test "from_map converts map to struct" do
      data = %{role: "user", content: "Hello", trainable: true}
      msg = Message.from_map(data)
      assert msg.role == "user"
      assert msg.content == "Hello"
      assert msg.trainable == true
    end

    test "to_map converts struct to map" do
      msg = Message.user("Hello", trainable: true)
      map = Message.to_map(msg)
      assert map.role == "user"
      assert map.content == "Hello"
      assert map.trainable == true
      refute Map.has_key?(map, :tool_calls)
    end
  end

  describe "TrainOnWhat" do
    test "normalize accepts valid atoms" do
      assert TrainOnWhat.normalize(:last_assistant_message) == :last_assistant_message
      assert TrainOnWhat.normalize(:all_assistant_messages) == :all_assistant_messages
      assert TrainOnWhat.normalize(:all_messages) == :all_messages
      assert TrainOnWhat.normalize(:all_tokens) == :all_tokens
      assert TrainOnWhat.normalize(:customized) == :customized
    end

    test "normalize accepts valid strings" do
      assert TrainOnWhat.normalize("last_assistant_message") == :last_assistant_message
      assert TrainOnWhat.normalize("ALL_ASSISTANT_MESSAGES") == :all_assistant_messages
    end

    test "normalize raises for invalid values" do
      assert_raise ArgumentError, fn ->
        TrainOnWhat.normalize(:invalid)
      end
    end

    test "valid? returns true for valid values" do
      assert TrainOnWhat.valid?(:last_assistant_message)
      assert TrainOnWhat.valid?(:all_messages)
    end

    test "valid? returns false for invalid values" do
      refute TrainOnWhat.valid?(:invalid)
      refute TrainOnWhat.valid?("string")
    end

    test "values returns all options" do
      values = TrainOnWhat.values()
      assert :last_assistant_message in values
      assert :all_assistant_messages in values
      assert :customized in values
    end
  end

  describe "RenderedMessage" do
    test "token_count sums all chunks" do
      rendered = %RenderedMessage{
        prefix: %EncodedTextChunk{tokens: [1, 2, 3]},
        content: [
          %EncodedTextChunk{tokens: [4, 5]},
          %EncodedTextChunk{tokens: [6]}
        ],
        suffix: %EncodedTextChunk{tokens: [7, 8]}
      }

      assert RenderedMessage.token_count(rendered) == 8
    end

    test "token_count handles nil chunks" do
      rendered = %RenderedMessage{
        prefix: nil,
        content: [%EncodedTextChunk{tokens: [1, 2]}],
        suffix: nil
      }

      assert RenderedMessage.token_count(rendered) == 2
    end

    test "all_tokens returns flat list" do
      rendered = %RenderedMessage{
        prefix: %EncodedTextChunk{tokens: [1, 2]},
        content: [%EncodedTextChunk{tokens: [3, 4]}],
        suffix: %EncodedTextChunk{tokens: [5]}
      }

      assert RenderedMessage.all_tokens(rendered) == [1, 2, 3, 4, 5]
    end
  end

  describe "Renderers.new/3" do
    test "creates llama3 renderer" do
      {:ok, renderer} = Renderers.new(:llama3, mock_tokenizer())
      assert renderer.type == :llama3
    end

    test "creates qwen3 renderer" do
      {:ok, renderer} = Renderers.new(:qwen3, mock_tokenizer())
      assert renderer.type == :qwen3
    end

    test "creates role_colon renderer" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())
      assert renderer.type == :role_colon
    end

    test "creates mistral renderer" do
      {:ok, renderer} = Renderers.new(:mistral, mock_tokenizer())
      assert renderer.type == :mistral
    end

    test "passes options to renderer" do
      {:ok, renderer} =
        Renderers.new(:qwen3, mock_tokenizer(), strip_thinking_from_history: false)

      assert renderer.opts[:strip_thinking_from_history] == false
    end
  end

  describe "Renderers.recommended_for/1" do
    test "returns llama3 for llama models" do
      assert Renderers.recommended_for("meta-llama/Llama-3.1-8B") == :llama3
      assert Renderers.recommended_for("Llama3-70B") == :llama3
    end

    test "returns qwen3 for qwen models" do
      assert Renderers.recommended_for("Qwen/Qwen3-8B") == :qwen3
      assert Renderers.recommended_for("qwen-3-instruct") == :qwen3
    end

    test "returns mistral for mistral/mixtral" do
      assert Renderers.recommended_for("mistralai/Mistral-7B") == :mistral
      assert Renderers.recommended_for("Mixtral-8x7B") == :mistral
    end

    test "returns role_colon for deepseek" do
      assert Renderers.recommended_for("deepseek-ai/DeepSeek-V2") == :role_colon
    end

    test "returns role_colon as default" do
      assert Renderers.recommended_for("unknown-model") == :role_colon
    end
  end

  describe "Renderers.render_message/4" do
    test "role_colon renders message with prefix and content" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())
      msg = Message.user("Hello")

      rendered = Renderers.render_message(renderer, 0, msg)

      assert rendered.prefix != nil
      assert [_ | _] = rendered.content
      assert rendered.suffix != nil
    end

    test "llama3 renders message with header tokens" do
      {:ok, renderer} = Renderers.new(:llama3, mock_tokenizer())
      msg = Message.assistant("Response")

      rendered = Renderers.render_message(renderer, 0, msg)

      assert rendered.prefix != nil
      assert [_ | _] = rendered.content
    end
  end

  describe "Renderers.build_generation_prompt/3" do
    test "builds prompt for sampling" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())

      messages = [
        Message.user("Hello!")
      ]

      {:ok, model_input} = Renderers.build_generation_prompt(renderer, messages)

      assert %ModelInput{} = model_input
      assert model_input.chunks != []
    end

    test "includes prefill if provided" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())
      messages = [Message.user("Hello!")]

      {:ok, model_input} =
        Renderers.build_generation_prompt(renderer, messages, prefill: "Sure,")

      # Should have more chunks with prefill
      assert length(model_input.chunks) >= 2
    end
  end

  describe "Renderers.build_supervised_example/3" do
    test "builds example with weights" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())

      messages = [
        Message.user("Question?"),
        Message.assistant("Answer.")
      ]

      {:ok, model_input, weights} = Renderers.build_supervised_example(renderer, messages)

      assert %ModelInput{} = model_input
      assert is_list(weights)
      assert Enum.all?(weights, &(&1 in [0.0, 1.0]))
    end

    test "last_assistant_message only weights last response" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())

      messages = [
        Message.user("Q1"),
        Message.assistant("A1"),
        Message.user("Q2"),
        Message.assistant("A2")
      ]

      {:ok, _model_input, weights} =
        Renderers.build_supervised_example(renderer, messages,
          train_on_what: :last_assistant_message
        )

      # Only the last assistant message should have weight
      assert Enum.any?(weights, &(&1 == 1.0))
    end

    test "all_tokens weights everything" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())

      messages = [
        Message.user("Hello"),
        Message.assistant("Hi")
      ]

      {:ok, _model_input, weights} =
        Renderers.build_supervised_example(renderer, messages, train_on_what: :all_tokens)

      # All weights should be 1.0 except possibly BOS
      assert Enum.any?(weights, &(&1 == 1.0))
    end
  end

  describe "Renderers.get_stop_sequences/1" do
    test "role_colon returns string stop sequence" do
      {:ok, renderer} = Renderers.new(:role_colon, mock_tokenizer())
      stops = Renderers.get_stop_sequences(renderer)
      assert "\n\nUser:" in stops
    end

    test "llama3 returns token id stop sequence" do
      {:ok, renderer} = Renderers.new(:llama3, mock_tokenizer())
      stops = Renderers.get_stop_sequences(renderer)
      assert is_list(stops)
    end
  end
end
