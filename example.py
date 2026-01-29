import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from racc.monkeypatch import replace_llama, replace_mistral

os.environ["CUDA_VISIBLE_DEVICES"] = "0"                # gpu count - 1
use_racc = True                                        # use_racc enables only compression
use_index = True                                        # use_index enables the recall mechanism
window_compress_ratio = 0.6                             # window_compress_ratio range [0.0 - 1.0], recommended value: 0.6
use_parallel = False                                    # if build index by parallel
index_mode = "flat"                                     # [flat, ivfpq, hnsw]
max_new_token = 2000                                    # recommended value > 1000
model_path = ""                                         # model path

if use_parallel == True:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
if use_racc == True:
    if "llama" in model_path.lower():
        replace_llama(use_index, window_compress_ratio, index_mode)
    if "mistral" in model_path.lower():
        replace_mistral(use_index, window_compress_ratio, index_mode)

# loading model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# create pipeline
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# input 
question = """
The neon hum of the 
"""

start_time = time.time()
response = qa_pipeline(question, max_new_tokens=max_new_token, num_return_sequences=1, do_sample=False)
end_time = time.time()

generated_text = response[0]["generated_text"]
num_tokens = len(tokenizer.encode(generated_text))
generation_time = end_time - start_time

print("--------------------------------------------------------------------------------")
print(f"Question: {question}")
print(f"Answer: {generated_text}")
print(f"Tokens generated: {num_tokens}")
print(f"Time taken: {generation_time:.2f} seconds")
print("--------------------------------------------------------------------------------")
