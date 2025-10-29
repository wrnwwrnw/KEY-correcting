import torch
import os
import inspect
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    TransformersKwargs,
    auto_docstring, 
    can_return_tuple,
    logging,
)
from transformers.masking_utils import create_masks_for_generate
from transformers.pytorch_utils import isin_mps_friendly
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from racc.util import init_racckv
import math
from racc.index import flat_build, flat_search, ivfpq_build, ivfpq_search, hnsw_build, hnsw_search
import time
# from concurrent.futures import ProcessPoolExecutor
# import multiprocessing

logger = logging.get_logger(__name__)

build_map = {
    "flat": flat_build,
    "ivfpq": ivfpq_build,
    "hnsw": hnsw_build
}

search_map = {
    "flat": flat_search,
    "ivfpq": ivfpq_search,
    "hnsw": hnsw_search
}

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
    
def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:

    index_mode = os.getenv("INDEX_MODE", "flat").lower()
    build = build_map.get(index_mode, flat_build)
    search = search_map.get(index_mode, flat_search)
    
    if (
        "flash" in str(self.config._attn_implementation).lower()
        and os.getenv("USE_INDEX", "false").lower() == "true"
    ):
        #仅对flashattn实现了crcc
        init_racckv(self)
        #多次推理时，需要设置RESTART=1，重置缓存和索引
        if os.environ.get("RESTART") == "1":
            os.environ["RESTART"] = "0"
            del llama_attention_forward.cache
            del llama_attention_forward.cacheidx
            del llama_attention_forward.index_grid
            del llama_attention_forward.ifindex
        # 构造cpu容器、索引容器[层][头]、构建索引状态码,-1表示静止，0表示可以构建，1表示正在构造，2表示构造完成
        if not hasattr(llama_attention_forward, "cache"):
            llama_attention_forward.cache = DynamicCache()
        if not hasattr(llama_attention_forward, "cacheidx"):
            llama_attention_forward.cacheidx = [[] for _ in range(32)]
        if not hasattr(llama_attention_forward, "index_grid"):
            llama_attention_forward.index_grid = [[None for _ in range(32)] for _ in range(32)]
        if not hasattr(llama_attention_forward, "ifindex"):
        #    manager = multiprocessing.Manager()
        #    llama_attention_forward.ifindex = manager.list([-1])
            llama_attention_forward.ifindex = [-1]
        if llama_attention_forward.ifindex[0]==0 :
            #    executor = ProcessPoolExecutor()
            llama_attention_forward.ifindex[0] = 1 
            #    pool = Pool(10)
            #    result = pool.apply_async(build, (llama_attention_forward.cache, llama_attention_forward.index_grid, llama_attention_forward.ifindex))
            #    pool.close()
            #    future = executor.submit(build ,llama_attention_forward.cache ,llama_attention_forward.index_grid ,llama_attention_forward.ifindex)
            #    executor.shutdown(wait=False)
            build(llama_attention_forward.cache ,llama_attention_forward.index_grid ,llama_attention_forward.ifindex)
        bsz, q_len, _ = hidden_states.size()


        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        kv_seq_len = key_states.shape[-2] #新增变量，用于区分预填充阶段和decode阶段
        if past_key_values is not None:
            if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_values.get_seq_length(self.layer_idx)
            else:
                kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
            #预填充处理
                self.kv_seq_len = kv_seq_len # [SnapKV] register kv_seq_len
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)  
                key_states, value_states = past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                if self.layer_idx ==0:
                    llama_attention_forward.count1 = kv_seq_len
                    llama_attention_forward.count2 = 0
                    llama_attention_forward.conut3 = key_states_compress.shape[2]
            else:
                if self.layer_idx ==0:     
                    llama_attention_forward.count2 += 1
                self.kv_seq_len += q_len
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                # 核心代码
                if (llama_attention_forward.count2%200==0)and(llama_attention_forward.count2%400!=0)and(llama_attention_forward.ifindex[0]==2):
                    search(query_states ,self.layer_idx ,0 ,int(self.kv_seq_len / 100) ,llama_attention_forward.cacheidx ,llama_attention_forward.index_grid ,llama_attention_forward.cache)
                # 查询
                if llama_attention_forward.count2>360:
                    # 创建表格存储注意力打分
                    if hasattr(llama_attention_forward, "sheet") and llama_attention_forward.count2 == 361 and self.layer_idx ==0:
                        del llama_attention_forward.sheet
                    if not hasattr(llama_attention_forward, "sheet"):
                        llama_attention_forward.sheet = torch.zeros(32, 40, 32, key_states.shape[2]-llama_attention_forward.conut3-1)
                    head_dim = 128
                    attn_weight = torch.matmul(query_states, key_states[:, :, llama_attention_forward.conut3:llama_attention_forward.conut3+llama_attention_forward.sheet.shape[3], :].transpose(2, 3)) / math.sqrt(head_dim)
                    attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    llama_attention_forward.sheet[self.layer_idx, llama_attention_forward.count2-361, :, :] = attn_weight.squeeze(0).squeeze(1)
                    # 这段有点问题，如果满了400个进剪枝逻辑
                    if llama_attention_forward.count2==400:
                        key_states_compress, value_states_compress = self.kv_cluster.update_kv_decode(key_states, value_states, llama_attention_forward.sheet, self.layer_idx, llama_attention_forward.cache, llama_attention_forward.conut3, self.kv_seq_len, cache_kwargs,llama_attention_forward.ifindex, llama_attention_forward.cacheidx)
                        past_key_values.layers[self.layer_idx].keys = key_states_compress
                        past_key_values.layers[self.layer_idx].values = value_states_compress
                        #这次修剪结束，重置计数单位
                        if self.layer_idx == 31:
                            llama_attention_forward.count2 = 0
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

    elif (
        "flash" in str(self.config._attn_implementation).lower()
        and os.getenv("USE_INDEX", "false").lower() == "false"
    ):  
        #仅压缩处理，不使用index
        init_racckv(self)
        bsz, q_len, _ = hidden_states.size()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        kv_seq_len = key_states.shape[-2] #新增变量，用于区分预填充阶段和decode阶段
        if past_key_values is not None:
            if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_values.get_seq_length(self.layer_idx)
            else:
                kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
            #预填充处理
                self.kv_seq_len = kv_seq_len # [SnapKV] register kv_seq_len
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)  
                key_states, value_states = past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                if self.layer_idx ==0:
                    llama_attention_forward.count1 = kv_seq_len
                    llama_attention_forward.count2 = 0
                    llama_attention_forward.conut3 = key_states_compress.shape[2]
            else:
                if self.layer_idx ==0:     
                    llama_attention_forward.count2 += 1
                self.kv_seq_len += q_len
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                if llama_attention_forward.count2>360:
                    # 创建表格存储注意力打分，用于后续压缩
                    if hasattr(llama_attention_forward, "sheet") and llama_attention_forward.count2 == 361 and self.layer_idx ==0:
                        del llama_attention_forward.sheet
                    if not hasattr(llama_attention_forward, "sheet"):
                        llama_attention_forward.sheet = torch.zeros(32, 40, 32, key_states.shape[2]-llama_attention_forward.conut3-1)
                    head_dim = 128
                    attn_weight = torch.matmul(query_states, key_states[:, :, llama_attention_forward.conut3:llama_attention_forward.conut3+llama_attention_forward.sheet.shape[3], :].transpose(2, 3)) / math.sqrt(head_dim)
                    attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    
                    llama_attention_forward.sheet[self.layer_idx, llama_attention_forward.count2-361, :, :] = attn_weight.squeeze(0).squeeze(1)
                    # 这段有点问题，如果满了400个进剪枝逻辑
                    if llama_attention_forward.count2==400:
                        key_states_compress, value_states_compress = self.kv_cluster.update_kv_decode_concise(key_states, value_states, llama_attention_forward.sheet, self.layer_idx, llama_attention_forward.conut3, self.kv_seq_len, cache_kwargs)
                        past_key_values.layers[self.layer_idx].keys = key_states_compress
                        past_key_values.layers[self.layer_idx].values = value_states_compress
                        #这次修剪结束，重置计数单位
                        if self.layer_idx == 31:
                            llama_attention_forward.count2 = 0
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

    else:
        #其他注意力模式
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

def prepare_inputs_for_generation_llama(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):

        if past_key_values is None: # [SnapKV]
            for layer in self.model.layers:
                layer.self_attn.kv_seq_len = 0
        """
        Prepare the model inputs for generation. Notable steps include selecting the correct input key and cloning when appropriate,
        creating position_ids from the attention_mask when missing, slicing inputs and converting 2D attention masks to 4D for
        compilable caches, and finally forwarding all additional keyword arguments unchanged to the model's forward pass.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        # We check `use_cache` below because some stateful models (like `recurrent_gemma`) expect input slicing if
        # their caching mechanism is used. To define `use_cache`, the user-defined argument takes precedence.
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)
        if past_key_values is None or use_cache:
            # TODO (joao): handle the case where cache length == input_ids length. The function below results in an
            # exception because we get empty input_ids after slicing. In essence, we need to roll back the cache 1
            # token to recompute the logits for the first token to be generated (but not all caches support roll backs)
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None or use_cache:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
        # pass)
        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            # If it's not defined, it means the model uses the new general mask API
            if causal_mask_creation_function is None:  # can't be found
                token_type_ids = model_inputs.get("token_type_ids")
                position_ids = model_inputs.get(position_ids_key)
                # Some models may overwrite the general one
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask = causal_mask_creation_function(
                    config=self.config,
                    # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs