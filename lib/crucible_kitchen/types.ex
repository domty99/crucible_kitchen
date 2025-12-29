defmodule CrucibleKitchen.Types do
  @moduledoc """
  Core data types for CrucibleKitchen.

  This module re-exports types from CrucibleTrain for convenience,
  so users only need to interact with CrucibleKitchen.

  ## Available Types

  - `Datum` - A single training example with model input and loss inputs
  - `ModelInput` - Input to the model (chunks of tokens)
  - `EncodedTextChunk` - A chunk of encoded text tokens
  - `TensorData` - Raw tensor data with shape and dtype

  ## Usage

      alias CrucibleKitchen.Types.{Datum, ModelInput, EncodedTextChunk}

      datum = %Datum{
        model_input: %ModelInput{
          chunks: [%EncodedTextChunk{tokens: [1, 2, 3]}]
        },
        loss_fn_inputs: %{
          labels: %TensorData{data: <<...>>, shape: [3], dtype: :s32}
        }
      }
  """

  # Re-export CrucibleTrain types
  # This allows users to use CrucibleKitchen.Types.Datum instead of CrucibleTrain.Types.Datum

  defmodule Datum do
    @moduledoc """
    A single training example.

    Contains the model input (tokenized text) and loss function inputs
    (labels, masks, etc.).
    """

    alias CrucibleKitchen.Types.ModelInput
    alias CrucibleKitchen.Types.TensorData

    @type t :: %__MODULE__{
            model_input: ModelInput.t(),
            loss_fn_inputs: %{atom() => TensorData.t()}
          }

    defstruct [:model_input, :loss_fn_inputs]

    @doc """
    Create a Datum from CrucibleTrain.Types.Datum.
    """
    def from_crucible_train(%CrucibleTrain.Types.Datum{} = d) do
      %__MODULE__{
        model_input: ModelInput.from_crucible_train(d.model_input),
        loss_fn_inputs:
          d.loss_fn_inputs
          |> Enum.map(fn {k, v} ->
            {k, TensorData.from_crucible_train(v)}
          end)
          |> Map.new()
      }
    end

    @doc """
    Convert to CrucibleTrain.Types.Datum.
    """
    def to_crucible_train(%__MODULE__{} = d) do
      %CrucibleTrain.Types.Datum{
        model_input: ModelInput.to_crucible_train(d.model_input),
        loss_fn_inputs:
          d.loss_fn_inputs
          |> Enum.map(fn {k, v} ->
            {k, TensorData.to_crucible_train(v)}
          end)
          |> Map.new()
      }
    end
  end

  defmodule ModelInput do
    @moduledoc """
    Model input consisting of chunks of encoded text.
    """

    alias CrucibleKitchen.Types.EncodedTextChunk

    @type t :: %__MODULE__{
            chunks: [EncodedTextChunk.t()]
          }

    defstruct chunks: []

    def from_crucible_train(%CrucibleTrain.Types.ModelInput{} = mi) do
      %__MODULE__{
        chunks: Enum.map(mi.chunks, &EncodedTextChunk.from_crucible_train/1)
      }
    end

    def to_crucible_train(%__MODULE__{} = mi) do
      %CrucibleTrain.Types.ModelInput{
        chunks: Enum.map(mi.chunks, &EncodedTextChunk.to_crucible_train/1)
      }
    end

    @doc """
    Get all tokens as a flat list.
    """
    @spec all_tokens(t()) :: [non_neg_integer()]
    def all_tokens(%__MODULE__{chunks: chunks}) do
      chunks
      |> Enum.flat_map(& &1.tokens)
    end
  end

  defmodule EncodedTextChunk do
    @moduledoc """
    A chunk of encoded (tokenized) text.
    """

    @type t :: %__MODULE__{
            tokens: [non_neg_integer()]
          }

    defstruct tokens: []

    def from_crucible_train(%CrucibleTrain.Types.EncodedTextChunk{} = c) do
      %__MODULE__{tokens: c.tokens}
    end

    def to_crucible_train(%__MODULE__{} = c) do
      %CrucibleTrain.Types.EncodedTextChunk{tokens: c.tokens}
    end
  end

  defmodule TensorData do
    @moduledoc """
    Raw tensor data with shape and dtype.
    """

    @type dtype :: :f32 | :f16 | :bf16 | :s32 | :s64 | :u8 | :u32
    @type t :: %__MODULE__{
            data: binary(),
            shape: [non_neg_integer()],
            dtype: dtype()
          }

    defstruct [:data, :shape, :dtype]

    def from_crucible_train(%CrucibleTrain.Types.TensorData{} = td) do
      %__MODULE__{data: td.data, shape: td.shape, dtype: td.dtype}
    end

    def to_crucible_train(%__MODULE__{} = td) do
      %CrucibleTrain.Types.TensorData{data: td.data, shape: td.shape, dtype: td.dtype}
    end
  end
end
