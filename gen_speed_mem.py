from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from kvpress import SnapKVPress, QFilterPress, StreamingLLMPress, ExpectedAttentionPress, KnormPress
from utils import QFilters



device = "cuda:0"
ckpt = "meta-llama/Llama-3.1-8B"
# n_tokens = 128_000
n_tokens = 32_000
compression_ratio = 0.75

HF_TOKEN = os.environ["HF_TOKEN"]
tokenizer = AutoTokenizer.from_pretrained(ckpt, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(ckpt, token=HF_TOKEN, attn_implementation="flash_attention_2", device_map="auto",  low_cpu_mem_usage=True, torch_dtype="bfloat16")


dataset = load_dataset("NeelNanda/pile-10k", split="train[1000:]")

model_suffix = ckpt.split("/")[-1]
q_filters = QFilters.from_pretrained(f"../filters/{model_suffix}_qfilt")
q_filters = q_filters.q_filters.to(device)


q_press = QFilterPress(compression_ratio)
q_press.q_filters = q_filters

expattn_press = ExpectedAttentionPress(compression_ratio)

def get_generation_stats(press, n_tokens, max_new_tokens = 100, inputs = None, cache_implementation="dynamic"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", attn_implementation="flash_attention_2").to(device)
    # disable EosTokenCriteria stopping criteria
    model.generation_config.eos_token_id = None
    model.generation_config.stop_strings = None
    
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    initial_peak_memory = torch.cuda.max_memory_allocated()
    
    if inputs is None:
        inputs = torch.arange(n_tokens).reshape([1, n_tokens])
    inputs = inputs.to(device)

    with torch.no_grad(), press(model):
        kwargs = dict()
        if cache_implementation == "quantized":
            kwargs = dict(cache_implementation="quantized",
                          cache_config={"backend": "quanto", "nbits": 4})
            
        start = time()
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens,
                                 generation_config=model.generation_config,
                                 pad_token_id=-1, **kwargs)
        torch.cuda.current_stream().synchronize()
        total_time = time() - start
        assert outputs.shape == (1, n_tokens + max_new_tokens), outputs.shape

    peak_memory = torch.cuda.max_memory_allocated()

    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"Idle Peak memory": idle_peak_memory / 1024**3,
            "Initial Peak memory": initial_peak_memory / 1024**3,
            "Total time": total_time, 
            "Peak memory usage": peak_memory / 1024**3
           }

#warmup
stats = get_generation_stats(q_press, n_tokens, max_new_tokens = 2)

stats = get_generation_stats(expattn_press, n_tokens, max_new_tokens = 16)
print(stats)

stats = get_generation_stats(q_press, n_tokens, max_new_tokens = 16)
print(stats)
