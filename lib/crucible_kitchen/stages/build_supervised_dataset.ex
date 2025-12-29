defmodule CrucibleKitchen.Stages.BuildSupervisedDataset do
  @moduledoc """
  Stage for building a supervised training dataset.

  Takes raw samples from the dataset store and builds a
  CrucibleTrain.Supervised.Dataset for training.

  ## State Requirements

  - `:dataset_handle` - Dataset handle from LoadDataset
  - `:tokenizer` - Tokenizer from InitTokenizer

  ## Configuration

  - `:model` - Model name (for renderer selection)
  - `:batch_size` - Training batch size
  - `:max_length` - Maximum sequence length
  - `:train_on` - What parts to train on (for masked loss)

  ## State Updates

  - `:dataset` - Built supervised dataset
  - `:num_batches` - Number of batches
  - `:total_steps` - Total training steps (epochs * batches)
  """

  use CrucibleKitchen.Stage

  alias CrucibleTrain.Ports.DatasetStore
  alias CrucibleTrain.Renderers.{Renderer, TrainOnWhat, Types}
  alias CrucibleTrain.Supervised.{Common, DatasetFromSamples}
  alias CrucibleTrain.Supervised.Dataset, as: SupervisedDataset

  require Logger

  @impl true
  def name, do: :build_supervised_dataset

  @impl true
  def execute(context) do
    dataset_handle = get_state(context, :dataset_handle)
    tokenizer = get_state(context, :tokenizer)

    model = get_config(context, :model)
    batch_size = get_config(context, :batch_size, 128)
    max_length = get_config(context, :max_length, 32_768)
    num_epochs = get_config(context, :epochs, 1)
    train_on = get_config(context, :train_on, :last_assistant_message)

    Logger.info("[BuildSupervisedDataset] Building dataset")
    Logger.debug("[BuildSupervisedDataset] batch_size=#{batch_size}, max_length=#{max_length}")

    ports = get_train_ports(context)

    case DatasetStore.to_list(ports, dataset_handle) do
      {:ok, samples} ->
        Logger.info("[BuildSupervisedDataset] Loaded #{length(samples)} samples")

        # Build supervised dataset using CrucibleTrain if available
        dataset = build_dataset(samples, tokenizer, model, batch_size, max_length, train_on)
        num_batches = get_num_batches(dataset)
        total_steps = num_batches * num_epochs

        Logger.info("[BuildSupervisedDataset] #{num_batches} batches, #{total_steps} total steps")

        context =
          context
          |> put_state(:dataset, dataset)
          |> put_state(:num_batches, num_batches)
          |> put_state(:total_steps, total_steps)
          |> record_metric(:samples_loaded, length(samples))
          |> record_metric(:num_batches, num_batches)

        {:ok, context}

      {:error, reason} ->
        Logger.error("[BuildSupervisedDataset] Failed to load samples: #{inspect(reason)}")
        {:error, {:dataset_to_list_failed, reason}}
    end
  end

  defp build_dataset(samples, tokenizer, model, batch_size, max_length, train_on) do
    renderer_module = get_renderer(model)

    case init_renderer(renderer_module, tokenizer) do
      {:ok, renderer_state} ->
        dataset_from_samples(
          samples,
          batch_size,
          max_length,
          train_on,
          renderer_module,
          renderer_state
        )

      {:error, _reason} ->
        fallback_dataset(samples, batch_size)
    end
  end

  defp get_num_batches(dataset) do
    cond do
      Map.has_key?(dataset, :num_batches) ->
        dataset.num_batches

      Map.has_key?(dataset, :batches) ->
        length(dataset.batches)

      match?(%{__struct__: _}, dataset) and
          function_exported?(SupervisedDataset, :length, 1) ->
        SupervisedDataset.length(dataset)

      true ->
        0
    end
  end

  defp get_renderer(model_name) when is_binary(model_name) do
    cond do
      String.contains?(model_name, "Llama-3") -> CrucibleTrain.Renderers.Llama3
      String.contains?(model_name, "Qwen") -> CrucibleTrain.Renderers.Qwen3
      String.contains?(model_name, "DeepSeek") -> CrucibleTrain.Renderers.DeepSeekV3
      true -> CrucibleTrain.Renderers.RoleColon
    end
  end

  defp get_renderer(_), do: CrucibleTrain.Renderers.RoleColon

  defp fallback_dataset(samples, batch_size) do
    %{
      samples: samples,
      batches: Enum.chunk_every(samples, batch_size),
      num_batches: ceil(length(samples) / batch_size)
    }
  end

  defp init_renderer(_renderer_module, nil), do: {:error, :missing_tokenizer}
  defp init_renderer(renderer_module, tokenizer), do: renderer_module.init(tokenizer: tokenizer)

  defp dataset_from_samples(
         samples,
         batch_size,
         max_length,
         train_on,
         renderer_module,
         renderer_state
       ) do
    train_on_what = normalize_train_on_what(train_on)

    datum_builder = fn sample ->
      messages = sample_to_messages(sample)

      {model_input, weights} =
        Renderer.build_supervised_example(
          renderer_module,
          messages,
          train_on_what,
          renderer_state
        )

      Common.datum_from_model_input_weights(model_input, weights, max_length)
    end

    DatasetFromSamples.new(samples, batch_size, datum_builder)
  end

  defp sample_to_messages(%{"messages" => messages}), do: normalize_messages(messages)
  defp sample_to_messages(%{messages: messages}), do: normalize_messages(messages)

  defp sample_to_messages(%{"prompt" => prompt, "completion" => completion}) do
    [Types.message("user", prompt), Types.message("assistant", completion)]
  end

  defp sample_to_messages(%{prompt: prompt, completion: completion}) do
    [Types.message("user", prompt), Types.message("assistant", completion)]
  end

  defp sample_to_messages(%{"instruction" => instruction, "response" => response}) do
    [Types.message("user", instruction), Types.message("assistant", response)]
  end

  defp sample_to_messages(%{instruction: instruction, response: response}) do
    [Types.message("user", instruction), Types.message("assistant", response)]
  end

  defp sample_to_messages(%{"input" => input, "output" => output}) do
    [Types.message("user", input), Types.message("assistant", output)]
  end

  defp sample_to_messages(%{input: input, output: output}) do
    [Types.message("user", input), Types.message("assistant", output)]
  end

  defp sample_to_messages(sample) do
    raise ArgumentError, "Unsupported sample format: #{inspect(sample)}"
  end

  defp normalize_messages(messages) when is_list(messages) do
    Enum.map(messages, &normalize_message/1)
  end

  defp normalize_message(%Types.Message{} = message), do: message

  defp normalize_message(%{"role" => role, "content" => content} = message) do
    Types.message(role, content,
      trainable: Map.get(message, "trainable"),
      thinking: Map.get(message, "thinking"),
      tool_calls: Map.get(message, "tool_calls"),
      tool_call_id: Map.get(message, "tool_call_id"),
      name: Map.get(message, "name")
    )
  end

  defp normalize_message(%{role: role, content: content} = message) do
    Types.message(role, content,
      trainable: Map.get(message, :trainable),
      thinking: Map.get(message, :thinking),
      tool_calls: Map.get(message, :tool_calls),
      tool_call_id: Map.get(message, :tool_call_id),
      name: Map.get(message, :name)
    )
  end

  defp normalize_message(message) do
    raise ArgumentError, "Unsupported message format: #{inspect(message)}"
  end

  defp normalize_train_on_what(nil), do: TrainOnWhat.last_assistant_message()

  defp normalize_train_on_what(value) when is_atom(value) do
    normalize_train_on_what(Atom.to_string(value))
  end

  defp normalize_train_on_what(value) when is_binary(value) do
    case TrainOnWhat.from_string(value) do
      {:ok, normalized} -> normalized
      {:error, _} -> TrainOnWhat.last_assistant_message()
    end
  end

  defp normalize_train_on_what(_), do: TrainOnWhat.last_assistant_message()
end
