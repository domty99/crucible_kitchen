defmodule CrucibleKitchen.HyperparamTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Hyperparam

  describe "get_hidden_size/1" do
    test "returns exact match for known models" do
      assert Hyperparam.get_hidden_size("meta-llama/Llama-3.1-8B") == 4096
      assert Hyperparam.get_hidden_size("meta-llama/Llama-3.2-1B") == 2048
      assert Hyperparam.get_hidden_size("Qwen/Qwen3-8B") == 4096
      assert Hyperparam.get_hidden_size("mistralai/Mistral-7B-v0.1") == 4096
    end

    test "infers size from model family patterns" do
      # Should match llama-3.1-8b pattern
      assert Hyperparam.get_hidden_size("some-org/llama-3.1-8b-custom") == 4096
    end

    test "infers size from parameter count suffix" do
      assert Hyperparam.get_hidden_size("unknown/model-70b") == 8192
      assert Hyperparam.get_hidden_size("unknown/model-8b") == 4096
      assert Hyperparam.get_hidden_size("unknown/model-1b") == 2048
    end

    test "returns default for completely unknown models" do
      assert Hyperparam.get_hidden_size("completely-unknown-model") == 4096
    end
  end

  describe "get_model_family/1" do
    test "identifies llama models" do
      assert Hyperparam.get_model_family("meta-llama/Llama-3.1-8B") == :llama
      assert Hyperparam.get_model_family("Llama3-70B") == :llama
    end

    test "identifies qwen models" do
      assert Hyperparam.get_model_family("Qwen/Qwen3-8B") == :qwen
      assert Hyperparam.get_model_family("qwen-instruct") == :qwen
    end

    test "identifies mistral models" do
      assert Hyperparam.get_model_family("mistralai/Mistral-7B") == :mistral
      assert Hyperparam.get_model_family("Mixtral-8x7B") == :mistral
    end

    test "identifies deepseek models" do
      assert Hyperparam.get_model_family("deepseek-ai/DeepSeek-V2") == :deepseek
    end

    test "identifies kimi models" do
      assert Hyperparam.get_model_family("moonshotai/Kimi-K2-Thinking") == :kimi
    end

    test "returns unknown for unrecognized models" do
      assert Hyperparam.get_model_family("some-random-model") == :unknown
    end
  end

  describe "get_lr/2" do
    test "returns higher LR for LoRA by default" do
      lora_lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", is_lora: true)
      full_lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", is_lora: false)

      # LoRA should be ~10x higher
      assert_in_delta lora_lr / full_lr, 10.0, 0.01
    end

    test "defaults to LoRA mode" do
      lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B")
      lora_lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", is_lora: true)

      assert lr == lora_lr
    end

    test "scales LR based on model size" do
      # Smaller models should have higher LR
      lr_1b = Hyperparam.get_lr("meta-llama/Llama-3.2-1B")
      lr_8b = Hyperparam.get_lr("meta-llama/Llama-3.1-8B")

      assert lr_1b > lr_8b
    end

    test "accepts custom base_lr" do
      lr_default = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", base_lr: 5.0e-5)
      lr_custom = Hyperparam.get_lr("meta-llama/Llama-3.1-8B", base_lr: 1.0e-4)

      assert_in_delta lr_custom / lr_default, 2.0, 0.01
    end

    test "returns positive float" do
      lr = Hyperparam.get_lr("meta-llama/Llama-3.1-8B")
      assert is_float(lr)
      assert lr > 0
    end
  end

  describe "get_lora_lr_over_full_finetune_lr/2" do
    test "returns 10x multiplier" do
      assert Hyperparam.get_lora_lr_over_full_finetune_lr("any-model") == 10.0
    end
  end

  describe "get_lora_param_count/2" do
    test "returns positive integer" do
      count = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B")
      assert is_integer(count)
      assert count > 0
    end

    test "scales with rank" do
      count_16 = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B", rank: 16)
      count_32 = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B", rank: 32)

      # Double rank should double params
      assert_in_delta count_32 / count_16, 2.0, 0.01
    end

    test "larger models have more LoRA params" do
      count_1b = Hyperparam.get_lora_param_count("meta-llama/Llama-3.2-1B", rank: 32)
      count_8b = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B", rank: 32)

      assert count_8b > count_1b
    end

    test "defaults to rank 32" do
      count_default = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B")
      count_32 = Hyperparam.get_lora_param_count("meta-llama/Llama-3.1-8B", rank: 32)

      assert count_default == count_32
    end
  end

  describe "get_lora_lr_multiplier/1" do
    test "returns positive float" do
      multiplier = Hyperparam.get_lora_lr_multiplier("meta-llama/Llama-3.1-8B")
      assert is_float(multiplier)
      assert multiplier > 0
    end

    test "smaller models have higher multiplier" do
      mult_1b = Hyperparam.get_lora_lr_multiplier("meta-llama/Llama-3.2-1B")
      mult_8b = Hyperparam.get_lora_lr_multiplier("meta-llama/Llama-3.1-8B")

      assert mult_1b > mult_8b
    end
  end

  describe "get_recommended_batch_size/2" do
    test "returns positive integer" do
      batch = Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B")
      assert is_integer(batch)
      assert batch >= 1
    end

    test "LoRA allows larger batches" do
      lora_batch = Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B", is_lora: true)

      full_batch =
        Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B", is_lora: false)

      assert lora_batch >= full_batch
    end

    test "scales with GPU memory" do
      batch_24gb =
        Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B", gpu_memory_gb: 24)

      batch_48gb =
        Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B", gpu_memory_gb: 48)

      assert batch_48gb >= batch_24gb
    end

    test "smaller models allow larger batches" do
      batch_1b = Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.2-1B")
      batch_70b = Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-70B")

      assert batch_1b > batch_70b
    end

    test "longer sequences reduce batch size" do
      batch_2048 =
        Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B", sequence_length: 2048)

      batch_4096 =
        Hyperparam.get_recommended_batch_size("meta-llama/Llama-3.1-8B", sequence_length: 4096)

      assert batch_2048 >= batch_4096
    end
  end
end
