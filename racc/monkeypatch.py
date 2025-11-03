from importlib.metadata import version
import warnings
import transformers
import os
from racc.llamapatch import llama_attention_forward as llama_attention_forward_4_57, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_57
from racc.mistralpatch import mistral_attention_forward as mistral_attention_forward_4_57, prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_4_57

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version
   
def replace_llama(use_index, window_compress_ratio, index_mode):

    os.environ["USE_INDEX"] = str(use_index).lower()          # 'true' or 'false'
    os.environ["WINDOW_COMPRESS_RATIO"] = str(window_compress_ratio)  # e.g. '0.6'
    os.environ["INDEX_MODE"] = index_mode                     # e.g. 'flat\ivfpq\hnsw'

    transformers_version = check_version()
    version_list = ['4.57']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with RACC. RACC is tested with Transformers version {version_list}.")
    transformers.generation.GenerationMixin.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_4_57
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attention_forward_4_57

def replace_mistral(use_index, window_compress_ratio, index_mode):

    os.environ["USE_INDEX"] = str(use_index).lower()          # 'true' or 'false'
    os.environ["WINDOW_COMPRESS_RATIO"] = str(window_compress_ratio)  # e.g. '0.6'
    os.environ["INDEX_MODE"] = index_mode                     # e.g. 'flat\ivfpq\hnsw'

    transformers_version = check_version()
    version_list = ['4.57']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with RACC. RACC is tested with Transformers version {version_list}.")
    transformers.generation.GenerationMixin.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_4_57
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attention_forward_4_57
