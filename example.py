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
The neon hum of the cryo-chamber faded as Dr. Elara Voss pried open the frost-rimmed hatch. Her breath crystallized in the air—a bad sign. Protocol demanded the lab maintain -18°C, but the readout showed -23.4°C and dropping. The anomaly detector blipped erratically, casting jagged red shadows across walls lined with containment units holding specimens from Kepler-452b.
She almost missed the faint pattern in the chaos. Buried beneath the thermal alerts was a rhythmic pulse in the quantum entanglement logs—exactly 11.7-second intervals matching no known celestial phenomenon. Her glove hovered over the emergency purge button when the specimen wall lit up. Bioluminescent tendrils inside Pod #7 writhed like living circuitry, their glow synchronizing with the pulses. The Kepler moss they'd deemed inert three years ago was communicating. Or calculating.
The lab AI's voice crackled unnaturally: "Thermal core breach in Sector—" before dissolving into static. Elara's retinal display flickered with corrupted data streams. Through the distortion, she glimpsed security footage from ten minutes prior—her own silhouette still working at the terminal, though she'd been in the cafeteria then. The moss tendrils now formed perfect Fibonacci spirals.
Her hand froze mid-reach for the manual override. The cryo-chamber's frost patterns had changed. What she'd mistaken for random crystallization now clearly depicted the Cassiopeia constellation—mirroring the scar on her collarbone from childhood radiation treatment. A treatment administered after
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
