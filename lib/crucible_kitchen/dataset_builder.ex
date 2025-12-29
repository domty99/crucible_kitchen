defmodule CrucibleKitchen.DatasetBuilder do
  @moduledoc """
  Dataset building utilities for training and evaluation.

  This module provides a unified interface for loading and transforming
  datasets from various sources into training-ready formats.

  ## Sources

  - HuggingFace Hub (via hf_datasets_ex)
  - Local JSONL/JSON files
  - CSV files
  - In-memory lists

  ## Transformations

  - Chat formatting (system/user/assistant messages)
  - Prompt templating
  - Field extraction and mapping
  - Filtering and sampling

  ## Examples

      # Load from HuggingFace
      {:ok, dataset} = DatasetBuilder.from_hub("databricks/dolly-15k")

      # Load from JSONL file
      {:ok, dataset} = DatasetBuilder.from_jsonl("data/train.jsonl")

      # Transform to chat format
      chat_dataset = DatasetBuilder.to_chat(dataset, fn row ->
        [
          Message.user(row["instruction"]),
          Message.assistant(row["response"])
        ]
      end)

      # Tokenize and prepare for training
      {:ok, prepared} = DatasetBuilder.prepare_for_training(
        chat_dataset,
        renderer,
        max_length: 2048
      )
  """

  alias CrucibleKitchen.Renderers.Message
  alias CrucibleKitchen.Types.ModelInput

  @type row :: map()
  @type dataset :: [row()]

  @type source_opts :: [
          split: String.t(),
          subset: String.t() | nil,
          streaming: boolean(),
          cache_dir: String.t() | nil
        ]

  @type transform_opts :: [
          batch_size: pos_integer(),
          num_workers: pos_integer()
        ]

  # ==========================================================================
  # Loading from Sources
  # ==========================================================================

  @doc """
  Load a dataset from HuggingFace Hub.

  ## Parameters

  - `dataset_name` - HuggingFace dataset identifier (e.g., "tatsu-lab/alpaca")
  - `opts` - Options:
    - `:split` - Dataset split to load (default: "train")
    - `:subset` - Dataset subset/config name
    - `:cache_dir` - Local cache directory

  ## Returns

  - `{:ok, dataset}` - List of row maps
  - `{:error, reason}` - Loading failed
  """
  @spec from_hub(String.t(), source_opts()) :: {:ok, dataset()} | {:error, term()}
  def from_hub(dataset_name, opts \\ []) do
    split = Keyword.get(opts, :split, "train")
    subset = Keyword.get(opts, :subset)

    # Build dataset path
    path =
      if subset do
        "#{dataset_name}/#{subset}"
      else
        dataset_name
      end

    # Try to use HfDatasetsEx if available
    case Code.ensure_loaded(HfDatasetsEx) do
      {:module, _} ->
        load_with_hf_datasets_ex(path, split, opts)

      {:error, _} ->
        {:error, :hf_datasets_ex_not_available}
    end
  end

  @doc """
  Load a dataset from a JSONL file (one JSON object per line).
  """
  @spec from_jsonl(String.t()) :: {:ok, dataset()} | {:error, term()}
  def from_jsonl(path) do
    case File.read(path) do
      {:ok, content} ->
        rows =
          content
          |> String.split("\n", trim: true)
          |> Enum.map(&Jason.decode!/1)

        {:ok, rows}

      {:error, reason} ->
        {:error, {:file_read_failed, reason}}
    end
  rescue
    e -> {:error, {:json_parse_failed, Exception.message(e)}}
  end

  @doc """
  Load a dataset from a JSON file (array of objects).
  """
  @spec from_json(String.t()) :: {:ok, dataset()} | {:error, term()}
  def from_json(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, rows} when is_list(rows) -> {:ok, rows}
          {:ok, _} -> {:error, :expected_array}
          {:error, reason} -> {:error, {:json_parse_failed, reason}}
        end

      {:error, reason} ->
        {:error, {:file_read_failed, reason}}
    end
  end

  @doc """
  Load a dataset from a CSV file.
  """
  @spec from_csv(String.t(), keyword()) :: {:ok, dataset()} | {:error, term()}
  def from_csv(path, opts \\ []) do
    separator = Keyword.get(opts, :separator, ?,)
    headers = Keyword.get(opts, :headers, true)

    case File.read(path) do
      {:ok, content} ->
        rows =
          content
          |> String.split("\n", trim: true)
          |> parse_csv(separator, headers)

        {:ok, rows}

      {:error, reason} ->
        {:error, {:file_read_failed, reason}}
    end
  end

  @doc """
  Create a dataset from an in-memory list.
  """
  @spec from_list([map()]) :: {:ok, dataset()}
  def from_list(rows) when is_list(rows) do
    {:ok, rows}
  end

  # ==========================================================================
  # Transformations
  # ==========================================================================

  @doc """
  Transform dataset rows to chat message format.

  ## Parameters

  - `dataset` - The dataset to transform
  - `formatter` - Function that takes a row and returns list of Messages

  ## Returns

  Dataset with rows containing a "messages" key.
  """
  @spec to_chat(dataset(), (row() -> [Message.t()])) :: dataset()
  def to_chat(dataset, formatter) when is_function(formatter, 1) do
    Enum.map(dataset, fn row ->
      messages = formatter.(row)
      Map.put(row, "messages", messages)
    end)
  end

  @doc """
  Apply a transformation function to each row.
  """
  @spec map(dataset(), (row() -> row())) :: dataset()
  def map(dataset, transform_fn) do
    Enum.map(dataset, transform_fn)
  end

  @doc """
  Filter dataset rows.
  """
  @spec filter(dataset(), (row() -> boolean())) :: dataset()
  def filter(dataset, predicate) do
    Enum.filter(dataset, predicate)
  end

  @doc """
  Select specific fields from each row.
  """
  @spec select(dataset(), [String.t() | atom()]) :: dataset()
  def select(dataset, fields) do
    Enum.map(dataset, fn row ->
      Map.take(row, fields)
    end)
  end

  @doc """
  Rename fields in each row.
  """
  @spec rename(dataset(), %{String.t() => String.t()}) :: dataset()
  def rename(dataset, mappings) do
    Enum.map(dataset, &rename_row(&1, mappings))
  end

  defp rename_row(row, mappings) do
    Enum.reduce(mappings, row, fn {old_key, new_key}, acc ->
      case Map.pop(acc, old_key) do
        {nil, acc} -> acc
        {value, acc} -> Map.put(acc, new_key, value)
      end
    end)
  end

  @doc """
  Shuffle the dataset.
  """
  @spec shuffle(dataset(), integer() | nil) :: dataset()
  def shuffle(dataset, seed \\ nil) do
    if seed do
      :rand.seed(:exsss, {seed, seed, seed})
    end

    Enum.shuffle(dataset)
  end

  @doc """
  Take first n rows from dataset.
  """
  @spec take(dataset(), non_neg_integer()) :: dataset()
  def take(dataset, n) do
    Enum.take(dataset, n)
  end

  @doc """
  Sample n random rows from dataset.
  """
  @spec sample(dataset(), non_neg_integer(), integer() | nil) :: dataset()
  def sample(dataset, n, seed \\ nil) do
    dataset
    |> shuffle(seed)
    |> take(n)
  end

  @doc """
  Split dataset into train/val/test sets.

  ## Parameters

  - `dataset` - The dataset to split
  - `ratios` - Tuple of {train, val, test} ratios (must sum to 1.0)
  - `seed` - Random seed for shuffling

  ## Returns

  Map with :train, :val, :test keys.
  """
  @spec split(dataset(), {float(), float(), float()}, integer() | nil) :: %{
          train: dataset(),
          val: dataset(),
          test: dataset()
        }
  def split(dataset, ratios \\ {0.8, 0.1, 0.1}, seed \\ nil) do
    {train_ratio, val_ratio, _test_ratio} = ratios
    shuffled = shuffle(dataset, seed)
    total = length(shuffled)

    train_size = round(total * train_ratio)
    val_size = round(total * val_ratio)

    {train, rest} = Enum.split(shuffled, train_size)
    {val, test} = Enum.split(rest, val_size)

    %{train: train, val: val, test: test}
  end

  # ==========================================================================
  # Chat Format Helpers
  # ==========================================================================

  @doc """
  Create a standard instruction-following chat format.

  Takes a row with "instruction" and "response" (or "output") fields.
  """
  @spec instruction_chat_format(row(), keyword()) :: [Message.t()]
  def instruction_chat_format(row, opts \\ []) do
    {instruction, input, output} = extract_instruction_fields(row, opts)
    user_content = build_user_content(instruction, input)
    system_messages = build_system_messages(Keyword.get(opts, :system_prompt))
    system_messages ++ [Message.user(user_content), Message.assistant(output)]
  end

  defp extract_instruction_fields(row, opts) do
    instruction_key = Keyword.get(opts, :instruction_key, "instruction")
    input_key = Keyword.get(opts, :input_key, "input")
    output_key = Keyword.get(opts, :output_key, "output")

    instruction = row[instruction_key] || row["instruction"] || ""
    input = row[input_key] || row["input"] || ""
    output = row[output_key] || row["output"] || row["response"] || ""
    {instruction, input, output}
  end

  defp build_user_content(instruction, ""), do: instruction
  defp build_user_content(instruction, input), do: "#{instruction}\n\n#{input}"

  defp build_system_messages(nil), do: []
  defp build_system_messages(prompt), do: [Message.system(prompt)]

  @doc """
  Create a QA-style chat format.
  """
  @spec qa_chat_format(row(), keyword()) :: [Message.t()]
  def qa_chat_format(row, opts \\ []) do
    question_key = Keyword.get(opts, :question_key, "question")
    answer_key = Keyword.get(opts, :answer_key, "answer")
    context_key = Keyword.get(opts, :context_key, "context")

    question = row[question_key] || ""
    answer = row[answer_key] || ""
    context = row[context_key]

    user_content =
      if context do
        "Context: #{context}\n\nQuestion: #{question}"
      else
        question
      end

    [Message.user(user_content), Message.assistant(answer)]
  end

  # ==========================================================================
  # Preparation for Training
  # ==========================================================================

  @doc """
  Prepare dataset for training by tokenizing and creating attention masks.

  ## Parameters

  - `dataset` - Chat-formatted dataset (rows with "messages" key)
  - `renderer` - Renderer for tokenization
  - `opts` - Options:
    - `:max_length` - Maximum sequence length (default: 2048)
    - `:train_on_what` - Which tokens to train on (default: :last_assistant_message)

  ## Returns

  List of tokenized examples ready for training.
  """
  @spec prepare_for_training(dataset(), term(), keyword()) ::
          {:ok, [map()]} | {:error, term()}
  def prepare_for_training(dataset, renderer, opts \\ []) do
    max_length = Keyword.get(opts, :max_length, 2048)
    train_on_what = Keyword.get(opts, :train_on_what, :last_assistant_message)

    examples =
      dataset
      |> Enum.map(fn row ->
        messages = row["messages"]

        {:ok, model_input, weights} =
          CrucibleKitchen.Renderers.build_supervised_example(renderer, messages,
            train_on_what: train_on_what
          )

        tokens = ModelInput.all_tokens(model_input)

        # Truncate if needed
        if length(tokens) > max_length do
          %{
            tokens: Enum.take(tokens, max_length),
            weights: Enum.take(weights, max_length),
            truncated: true
          }
        else
          %{
            tokens: tokens,
            weights: weights,
            truncated: false
          }
        end
      end)

    {:ok, examples}
  end

  @doc """
  Get statistics about a dataset.
  """
  @spec stats(dataset()) :: map()
  def stats(dataset) do
    count = length(dataset)

    if count == 0 do
      %{count: 0, fields: [], sample: nil}
    else
      first = List.first(dataset)
      fields = Map.keys(first)

      %{
        count: count,
        fields: fields,
        sample: first
      }
    end
  end

  # ==========================================================================
  # Private Helpers
  # ==========================================================================

  defp load_with_hf_datasets_ex(path, split, _opts) do
    # This would be the actual HfDatasetsEx call
    # For now, return a placeholder since we're building the interface
    case HfDatasetsEx.load_dataset(path, split: split) do
      {:ok, dataset} -> {:ok, Enum.to_list(dataset)}
      error -> error
    end
  rescue
    e -> {:error, {:load_failed, Exception.message(e)}}
  end

  defp parse_csv(lines, separator, true = _headers) do
    case lines do
      [header_line | data_lines] ->
        headers =
          header_line
          |> String.split(<<separator>>)
          |> Enum.map(&String.trim/1)

        Enum.map(data_lines, fn line ->
          values =
            line
            |> String.split(<<separator>>)
            |> Enum.map(&String.trim/1)

          Enum.zip(headers, values) |> Map.new()
        end)

      [] ->
        []
    end
  end

  defp parse_csv(lines, separator, false = _headers) do
    Enum.map(lines, fn line ->
      line
      |> String.split(<<separator>>)
      |> Enum.map(&String.trim/1)
      |> Enum.with_index()
      |> Enum.map(fn {val, idx} -> {"col_#{idx}", val} end)
      |> Map.new()
    end)
  end
end
