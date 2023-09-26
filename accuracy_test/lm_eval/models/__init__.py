from . import llama, nakta

MODEL_REGISTRY = {
    # "hf": gpt2.HFLM,
    # "hf-causal": gpt2.HFLM,
    # "hf-causal-experimental": huggingface.AutoCausalLM,
    # "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    # "gpt2": gpt2.GPT2LM,
    # "gpt3": gpt3.GPT3LM,
    # "anthropic": anthropic_llms.AnthropicLM,
    # "textsynth": textsynth.TextSynthLM,
    # "dummy": dummy.DummyLM,
    "nakta": nakta.Nakta_ACC,
    "llama": llama.LLama_ACC,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
