import time

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from kvpress import SnapKVPress, QFilterPress, StreamingLLMPress, ExpectedAttentionPress, KnormPress
from utils import QFilters



device = "cuda:0"
ckpt = "meta-llama/Llama-3.1-8B"
n_tokens = 128_000
# n_tokens = 2048

HF_TOKEN = os.environ["HF_TOKEN"]
tokenizer = AutoTokenizer.from_pretrained(ckpt, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(ckpt, token=HF_TOKEN, attn_implementation="flash_attention_2", device_map="auto",  low_cpu_mem_usage=True, torch_dtype="bfloat16")

initial_peak_memory = torch.cuda.max_memory_allocated()
print(f"Initial peak memory usage: {initial_peak_memory / 1024**3:.2f} GB")


def get_size_of_cache(cache):
    value_cache = cache.value_cache
    key_cache = cache.key_cache
    size_in_bytes = 0
    for value in value_cache:
        size_in_bytes += value.element_size() * value.nelement()
    for key in key_cache:
        size_in_bytes += key.element_size() * key.nelement()
    return size_in_bytes

dataset = load_dataset("NeelNanda/pile-10k", split="train[1000:]")
NB_SAMPLES = 10

model_suffix = ckpt.split("/")[-1]
q_filters = QFilters.from_pretrained(f"../filters/{model_suffix}_qfilt")
q_filters = q_filters.q_filters.to(device)

inputs=None
for i, sample in tqdm(enumerate(dataset)):

    tokens = tokenizer(sample["text"], return_tensors="pt")
    if tokens.input_ids.shape[-1] < n_tokens:
        continue
    inputs = tokens.input_ids[:, :n_tokens].to(device)
    break

# Model warmup (for better prefilling time estimation)
with torch.no_grad():
    model(inputs)
torch.cuda.synchronize()

# Compute cache size and prefilling time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    prefilling_time = 0
    cache_size = None
    for _ in range(10):
        t0 = time.time_ns() / 1_000_000
        cache = model(inputs, num_logits_to_keep=1).past_key_values
        torch.cuda.current_stream().synchronize()
        t1 = time.time_ns() / 1_000_000
        prefilling_time += t1-t0
        if cache_size is None:
            cache_size = 2 * 2 * len(cache) * cache[0][0].numel() # 2 for keys and values, 2 for float16
        del cache
    prefilling_time /= 10

peak_memory = torch.cuda.max_memory_allocated()
torch.cuda.empty_cache()
print(f"Cache size for {n_tokens} tokens: {cache_size / 1024**3:.3f} GB")
print(f"Peak memory usage: {peak_memory / 1024**3:.2f} GB")
print(f"Peak memory w/o weights and KV cache: {(peak_memory - cache_size - initial_peak_memory) / 1024**3:.2f} GB")
print(f"Prefilling time: {prefilling_time:.2f}ms")

def timer(press, max_new_tokens = 1):
    t0 = time.time_ns() / 1_000_000

    with press(model):
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, pad_token_id=-1)
        assert outputs.shape == (1, n_tokens + max_new_tokens)
    torch.cuda.current_stream().synchronize()
    t1 = time.time_ns() / 1_000_000

    torch.cuda.empty_cache()
    torch.cuda.current_stream().synchronize()

    return t1-t0

# prefilling_time = 0

for compression_ratio in [0.9375]:
    print(f"\nCompression ratio: {compression_ratio}")
    print("SNAPKV")
    press = SnapKVPress(compression_ratio)
    torch.cuda.reset_peak_memory_stats()
    times = [] 
    for _ in tqdm(range(NB_SAMPLES)):
        times.append(timer(press))
    average_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    max_time = np.max(times)
    print(f"avg: {average_time:.2f}ms | std: {std_time:.2f}ms | median: {median_time:.2f}ms | max: {max_time:.2f}ms")
    print("-prefilling")
    print(f"avg: {average_time - prefilling_time:.2f}ms | median: {median_time - prefilling_time:.2f}ms | max: {max_time - prefilling_time:.2f}ms")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n\n")


    print("Knorm")
    press = KnormPress(compression_ratio)
    torch.cuda.reset_peak_memory_stats()
    times = [] 
    for _ in tqdm(range(NB_SAMPLES)):
        times.append(timer(press))
    average_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    max_time = np.max(times)
    print(f"avg: {average_time:.2f}ms | std: {std_time:.2f}ms | median: {median_time:.2f}ms | max: {max_time:.2f}ms")
    print("-prefilling")
    print(f"avg: {average_time - prefilling_time:.2f}ms | median: {median_time - prefilling_time:.2f}ms | max: {max_time - prefilling_time:.2f}ms")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n\n")

    print("QFILTER")
    press = QFilterPress(compression_ratio)
    press.q_filters = q_filters
    torch.cuda.reset_peak_memory_stats()
    times = [] 
    for _ in tqdm(range(NB_SAMPLES)):
        times.append(timer(press))
    average_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    max_time = np.max(times)
    print(f"avg: {average_time:.2f}ms | std: {std_time:.2f}ms | median: {median_time:.2f}ms | max: {max_time:.2f}ms")
    print("-prefilling")
    print(f"avg: {average_time - prefilling_time:.2f}ms | median: {median_time - prefilling_time:.2f}ms | max: {max_time - prefilling_time:.2f}ms")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n\n")

    print("Streaming-LLM")
    press = StreamingLLMPress(compression_ratio)
    torch.cuda.reset_peak_memory_stats()
    times = [] 
    for _ in tqdm(range(NB_SAMPLES)):
        times.append(timer(press))
    average_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    max_time = np.max(times)
    print(f"avg: {average_time:.2f}ms | std: {std_time:.2f}ms | median: {median_time:.2f}ms | max: {max_time:.2f}ms")
    print("-prefilling")
    print(f"avg: {average_time - prefilling_time:.2f}ms | median: {median_time - prefilling_time:.2f}ms | max: {max_time - prefilling_time:.2f}ms")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n\n")

    print("ExpectedAttn")
    press = ExpectedAttentionPress(compression_ratio)
    torch.cuda.reset_peak_memory_stats()
    times = [] 
    for _ in tqdm(range(NB_SAMPLES)):
        times.append(timer(press))
    average_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    max_time = np.max(times)
    print(f"avg: {average_time:.2f}ms | std: {std_time:.2f}ms | median: {median_time:.2f}ms | max: {max_time:.2f}ms")
    print("-prefilling")
    print(f"avg: {average_time - prefilling_time:.2f}ms | median: {median_time - prefilling_time:.2f}ms | max: {max_time - prefilling_time:.2f}ms")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n\n")


    