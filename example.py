import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from cdkey.monkeypatch.monkeypatch import replace_llama, replace_mistral

replace_llama()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_path = "*****"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

question = """
give me a list form 1890 to 2020 who win the nobel prices.
"""

start_time = time.time()
response = qa_pipeline(question, max_new_tokens = 1, num_return_sequences=1, do_sample=False)
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

