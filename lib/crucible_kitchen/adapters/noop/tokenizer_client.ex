defmodule CrucibleKitchen.Adapters.Noop.TokenizerClient do
  @moduledoc """
  Noop adapter for tokenizer helper (no port).

  Uses simple character-level tokenization. Useful for:
  - Testing stages in isolation
  - Development without tokenizer dependencies
  - Quick prototyping
  """

  @vocab_size 32_000
  @pad_token_id 0
  @eos_token_id 1
  @bos_token_id 2
  @unk_token_id 3

  def load(_opts, model_or_path, _load_opts) do
    tokenizer = %{
      type: :noop_char,
      model: model_or_path,
      vocab_size: @vocab_size,
      special_tokens: %{
        pad: @pad_token_id,
        eos: @eos_token_id,
        bos: @bos_token_id,
        unk: @unk_token_id
      }
    }

    {:ok, tokenizer}
  end

  def encode(_opts, _tokenizer, text, encode_opts) do
    add_special = Keyword.get(encode_opts, :add_special_tokens, true)
    max_length = Keyword.get(encode_opts, :max_length)

    # Simple character-level tokenization with offset for printable ASCII
    tokens =
      text
      |> String.to_charlist()
      |> Enum.map(fn char ->
        # Map to range [4, vocab_size) leaving room for special tokens
        id = rem(char, @vocab_size - 4) + 4
        min(id, @vocab_size - 1)
      end)

    tokens =
      if add_special do
        [@bos_token_id | tokens] ++ [@eos_token_id]
      else
        tokens
      end

    tokens =
      if max_length do
        Enum.take(tokens, max_length)
      else
        tokens
      end

    {:ok, tokens}
  end

  def decode(_opts, _tokenizer, token_ids, decode_opts) do
    skip_special = Keyword.get(decode_opts, :skip_special_tokens, true)

    special = [@pad_token_id, @eos_token_id, @bos_token_id, @unk_token_id]

    chars =
      token_ids
      |> Enum.reject(fn id ->
        skip_special and id in special
      end)
      |> Enum.map(fn id ->
        # Reverse the mapping
        char_code = rem(id - 4 + 32, 95) + 32
        <<char_code>>
      end)

    {:ok, Enum.join(chars)}
  end

  def encode_batch(_opts, tokenizer, texts, encode_opts) do
    results =
      Enum.map(texts, fn text ->
        {:ok, tokens} = encode([], tokenizer, text, encode_opts)
        tokens
      end)

    {:ok, results}
  end

  def get_vocab_size(_opts, _tokenizer) do
    {:ok, @vocab_size}
  end

  def get_special_tokens(_opts, tokenizer) do
    {:ok, tokenizer.special_tokens}
  end
end
