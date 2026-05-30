import torch
from torch import Tensor
from typing import Dict


def switch_backbone_key(key: str, value: Tensor) -> Dict[str, Tensor]:

    """
    Their keys 
    
    'transformer.h.0.ln_1.weight', 
    'transformer.h.0.ln_1.bias', 
    'transformer.h.0.attn.c_attn.weight', 
    'transformer.h.0.attn.c_attn.bias',
    'transformer.h.0.attn.c_proj.weight', 
    'transformer.h.0.attn.c_proj.bias',
    'transformer.h.0.ln_2.weight',
    'transformer.h.0.ln_2.bias',
    'transformer.h.0.mlp.c_fc.weight',
    'transformer.h.0.mlp.c_fc.bias',
    'transformer.h.0.mlp.c_proj.weight',
    'transformer.h.0.mlp.c_proj.bias',


    Our keys 

    'backbone.0.mlp.fc1.weight',
    'backbone.0.mlp.fc1.bias',
    'backbone.0.mlp.fc2.weight',
    'backbone.0.mlp.fc2.bias',
    'backbone.0.attention.W_k.weight',
    'backbone.0.attention.W_k.bias',
    'backbone.0.attention.W_q.weight',
    'backbone.0.attention.W_q.bias',
    'backbone.0.attention.W_v.weight',
    'backbone.0.attention.W_v.bias',
    'backbone.0.attention.W_o.weight',
    'backbone.0.attention.W_o.bias',
    'backbone.0.pre_layer_norm.weight',
    'backbone.0.pre_layer_norm.bias',
    'backbone.0.post_layer_norm.weight',
    'backbone.0.post_layer_norm.bias',
    """

    suffix_key_switcher = {
        "ln_1.weight": "pre_layer_norm.weight",
        "ln_1.bias": "pre_layer_norm.bias",
        "ln_2.weight": "post_layer_norm.weight",
        "ln_2.bias": "post_layer_norm.bias",
        "mlp.c_fc.weight": "mlp.fc1.weight",
        "mlp.c_fc.bias": "mlp.fc1.bias",
        "mlp.c_proj.weight": "mlp.fc2.weight",
        "mlp.c_proj.bias": "mlp.fc2.bias",
        "attn.c_proj.weight": "attention.W_o.weight",
        "attn.c_proj.bias": "attention.W_o.bias",
    }


    # backbone key in looks like transformer.h.0.attn.c_proj.weigh


    layer_num: str = key.split(".")[2]
    suffix: str = ".".join(key.split(".")[3:])

    if "c_attn.weight" in key:
        # We need to split the value up into three parts
        # Weight has shape n_dim, 

        # Remember that this was transposed
        _, embedding_dim = value.shape


        W_q, W_k, W_v = value.split(embedding_dim, dim=0)

        return {
            f"backbone.{layer_num}.attention.W_q.weight": W_q,
            f"backbone.{layer_num}.attention.W_k.weight": W_k,
            f"backbone.{layer_num}.attention.W_v.weight": W_v,
        }
    elif "c_attn.bias" in key:
        embedding_dim_times_3, = value.shape
        assert embedding_dim_times_3 % 3 == 0
        embedding_dim = embedding_dim_times_3 // 3

        b_q, b_k, b_v = value.split(embedding_dim, dim=0)

        return {
            f"backbone.{layer_num}.attention.W_q.bias": b_q,
            f"backbone.{layer_num}.attention.W_k.bias": b_k,
            f"backbone.{layer_num}.attention.W_v.bias": b_v,
        }
    else:

        return {
            f"backbone.{layer_num}.{suffix_key_switcher[suffix]}": value
        }

    


@torch.no_grad()
def state_dict_converter(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Convert the state dict from the HuggingFace GPT-2 model to the state dict of our Transformer model.
    """



    key_switcher = {
        'transformer.wte.weight': 'embeddings.weight', 
        'transformer.wpe.weight': 'position_embeddings.weight',
        'transformer.ln_f.weight': 'final_layer_norm.weight',
        'transformer.ln_f.bias': 'final_layer_norm.bias',
        'lm_head.weight': 'lm_head.weight'
    }

    new_state_dict = {}

    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for key, value in state_dict.items():

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        if any(key.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            #assert sd_hf[k].shape[::-1] == sd[k].shape
            value = value.t()

        if ".h." in key:
            # This is in the backbone
            new_state_dict = new_state_dict | switch_backbone_key(key, value.clone().detach())

        else:
            new_state_dict[key_switcher[key]] = value.clone().detach()

    return new_state_dict




"""
For reference, these are the keys of HF GPT-2 model:

'transformer.wte.weight', 
'transformer.wpe.weight', 
'transformer.h.0.ln_1.weight', 
'transformer.h.0.ln_1.bias', 
'transformer.h.0.attn.c_attn.weight', 
'transformer.h.0.attn.c_attn.bias',
'transformer.h.0.attn.c_proj.weight', 
'transformer.h.0.attn.c_proj.bias',
'transformer.h.0.ln_2.weight',
'transformer.h.0.ln_2.bias',
'transformer.h.0.mlp.c_fc.weight',
'transformer.h.0.mlp.c_fc.bias',
'transformer.h.0.mlp.c_proj.weight',
'transformer.h.0.mlp.c_proj.bias',
'transformer.h.1.ln_1.weight',
'transformer.h.1.ln_1.bias',
'transformer.h.1.attn.c_attn.weight',
'transformer.h.1.attn.c_attn.bias',
'transformer.h.1.attn.c_proj.weight',
'transformer.h.1.attn.c_proj.bias',
'transformer.h.1.ln_2.weight',
'transformer.h.1.ln_2.bias',
'transformer.h.1.mlp.c_fc.weight',
'transformer.h.1.mlp.c_fc.bias',
'transformer.h.1.mlp.c_proj.weight',
'transformer.h.1.mlp.c_proj.bias',
'transformer.h.2.ln_1.weight',
'transformer.h.2.ln_1.bias',
'transformer.h.2.attn.c_attn.weight',
'transformer.h.2.attn.c_attn.bias',
'transformer.h.2.attn.c_proj.weight',
'transformer.h.2.attn.c_proj.bias',
'transformer.h.2.ln_2.weight',
'transformer.h.2.ln_2.bias',
'transformer.h.2.mlp.c_fc.weight',
'transformer.h.2.mlp.c_fc.bias',
'transformer.h.2.mlp.c_proj.weight',
'transformer.h.2.mlp.c_proj.bias',
'transformer.h.3.ln_1.weight',
'transformer.h.3.ln_1.bias',
'transformer.h.3.attn.c_attn.weight',
'transformer.h.3.attn.c_attn.bias',
'transformer.h.3.attn.c_proj.weight',
'transformer.h.3.attn.c_proj.bias',
'transformer.h.3.ln_2.weight',
'transformer.h.3.ln_2.bias',
'transformer.h.3.mlp.c_fc.weight',
'transformer.h.3.mlp.c_fc.bias',
'transformer.h.3.mlp.c_proj.weight',
'transformer.h.3.mlp.c_proj.bias',
'transformer.h.4.ln_1.weight',
'transformer.h.4.ln_1.bias',
'transformer.h.4.attn.c_attn.weight',
'transformer.h.4.attn.c_attn.bias',
'transformer.h.4.attn.c_proj.weight',
'transformer.h.4.attn.c_proj.bias',
'transformer.h.4.ln_2.weight',
'transformer.h.4.ln_2.bias',
'transformer.h.4.mlp.c_fc.weight',
'transformer.h.4.mlp.c_fc.bias',
'transformer.h.4.mlp.c_proj.weight',
'transformer.h.4.mlp.c_proj.bias',
'transformer.h.5.ln_1.weight',
'transformer.h.5.ln_1.bias',
'transformer.h.5.attn.c_attn.weight',
'transformer.h.5.attn.c_attn.bias',
'transformer.h.5.attn.c_proj.weight',
'transformer.h.5.attn.c_proj.bias',
'transformer.h.5.ln_2.weight',
'transformer.h.5.ln_2.bias',
'transformer.h.5.mlp.c_fc.weight',
'transformer.h.5.mlp.c_fc.bias',
'transformer.h.5.mlp.c_proj.weight',
'transformer.h.5.mlp.c_proj.bias',
'transformer.h.6.ln_1.weight',
'transformer.h.6.ln_1.bias',
'transformer.h.6.attn.c_attn.weight',
'transformer.h.6.attn.c_attn.bias',
'transformer.h.6.attn.c_proj.weight',
'transformer.h.6.attn.c_proj.bias',
'transformer.h.6.ln_2.weight',
'transformer.h.6.ln_2.bias',
'transformer.h.6.mlp.c_fc.weight',
'transformer.h.6.mlp.c_fc.bias',
'transformer.h.6.mlp.c_proj.weight',
'transformer.h.6.mlp.c_proj.bias',
'transformer.h.7.ln_1.weight',
'transformer.h.7.ln_1.bias',
'transformer.h.7.attn.c_attn.weight',
'transformer.h.7.attn.c_attn.bias',
'transformer.h.7.attn.c_proj.weight',
'transformer.h.7.attn.c_proj.bias',
'transformer.h.7.ln_2.weight',
'transformer.h.7.ln_2.bias',
'transformer.h.7.mlp.c_fc.weight',
'transformer.h.7.mlp.c_fc.bias',
'transformer.h.7.mlp.c_proj.weight',
'transformer.h.7.mlp.c_proj.bias',
'transformer.h.8.ln_1.weight',
'transformer.h.8.ln_1.bias',
'transformer.h.8.attn.c_attn.weight',
'transformer.h.8.attn.c_attn.bias',
'transformer.h.8.attn.c_proj.weight',
'transformer.h.8.attn.c_proj.bias',
'transformer.h.8.ln_2.weight',
'transformer.h.8.ln_2.bias',
'transformer.h.8.mlp.c_fc.weight',
'transformer.h.8.mlp.c_fc.bias',
'transformer.h.8.mlp.c_proj.weight',
'transformer.h.8.mlp.c_proj.bias',
'transformer.h.9.ln_1.weight',
'transformer.h.9.ln_1.bias',
'transformer.h.9.attn.c_attn.weight',
'transformer.h.9.attn.c_attn.bias',
'transformer.h.9.attn.c_proj.weight',
'transformer.h.9.attn.c_proj.bias',
'transformer.h.9.ln_2.weight',
'transformer.h.9.ln_2.bias',
'transformer.h.9.mlp.c_fc.weight',
'transformer.h.9.mlp.c_fc.bias',
'transformer.h.9.mlp.c_proj.weight',
'transformer.h.9.mlp.c_proj.bias',
'transformer.h.10.ln_1.weight',
'transformer.h.10.ln_1.bias',
'transformer.h.10.attn.c_attn.weight',
'transformer.h.10.attn.c_attn.bias',
'transformer.h.10.attn.c_proj.weight',
'transformer.h.10.attn.c_proj.bias',
'transformer.h.10.ln_2.weight',
'transformer.h.10.ln_2.bias',
'transformer.h.10.mlp.c_fc.weight',
'transformer.h.10.mlp.c_fc.bias',
'transformer.h.10.mlp.c_proj.weight',
'transformer.h.10.mlp.c_proj.bias',
'transformer.h.11.ln_1.weight',
'transformer.h.11.ln_1.bias',
'transformer.h.11.attn.c_attn.weight',
'transformer.h.11.attn.c_attn.bias',
'transformer.h.11.attn.c_proj.weight',
'transformer.h.11.attn.c_proj.bias',
'transformer.h.11.ln_2.weight',
'transformer.h.11.ln_2.bias',
'transformer.h.11.mlp.c_fc.weight',
'transformer.h.11.mlp.c_fc.bias',
'transformer.h.11.mlp.c_proj.weight',
'transformer.h.11.mlp.c_proj.bias',
'transformer.ln_f.weight',
'transformer.ln_f.bias',
'lm_head.weight'



These our our keys:


'embeddings.weight',
'position_embeddings.weight',
'backbone.0.mlp.fc1.weight',
'backbone.0.mlp.fc1.bias',
'backbone.0.mlp.fc2.weight',
'backbone.0.mlp.fc2.bias',
'backbone.0.attention.W_k.weight',
'backbone.0.attention.W_k.bias',
'backbone.0.attention.W_q.weight',
'backbone.0.attention.W_q.bias',
'backbone.0.attention.W_v.weight',
'backbone.0.attention.W_v.bias',
'backbone.0.attention.W_o.weight',
'backbone.0.attention.W_o.bias',
'backbone.0.pre_layer_norm.weight',
'backbone.0.pre_layer_norm.bias',
'backbone.0.post_layer_norm.weight',
'backbone.0.post_layer_norm.bias',
'backbone.1.mlp.fc1.weight',
'backbone.1.mlp.fc1.bias',
'backbone.1.mlp.fc2.weight',
'backbone.1.mlp.fc2.bias',
'backbone.1.attention.W_k.weight',
'backbone.1.attention.W_k.bias',
'backbone.1.attention.W_q.weight',
'backbone.1.attention.W_q.bias',
'backbone.1.attention.W_v.weight',
'backbone.1.attention.W_v.bias',
'backbone.1.attention.W_o.weight',
'backbone.1.attention.W_o.bias',
'backbone.1.pre_layer_norm.weight',
'backbone.1.pre_layer_norm.bias',
'backbone.1.post_layer_norm.weight',
'backbone.1.post_layer_norm.bias',
'backbone.2.mlp.fc1.weight',
'backbone.2.mlp.fc1.bias',
'backbone.2.mlp.fc2.weight',
'backbone.2.mlp.fc2.bias',
'backbone.2.attention.W_k.weight',
'backbone.2.attention.W_k.bias',
'backbone.2.attention.W_q.weight',
'backbone.2.attention.W_q.bias',
'backbone.2.attention.W_v.weight',
'backbone.2.attention.W_v.bias',
'backbone.2.attention.W_o.weight',
'backbone.2.attention.W_o.bias',
'backbone.2.pre_layer_norm.weight',
'backbone.2.pre_layer_norm.bias',
'backbone.2.post_layer_norm.weight',
'backbone.2.post_layer_norm.bias',
'backbone.3.mlp.fc1.weight',
'backbone.3.mlp.fc1.bias',
'backbone.3.mlp.fc2.weight',
'backbone.3.mlp.fc2.bias',
'backbone.3.attention.W_k.weight',
'backbone.3.attention.W_k.bias',
'backbone.3.attention.W_q.weight',
'backbone.3.attention.W_q.bias',
'backbone.3.attention.W_v.weight',
'backbone.3.attention.W_v.bias',
'backbone.3.attention.W_o.weight',
'backbone.3.attention.W_o.bias',
'backbone.3.pre_layer_norm.weight',
'backbone.3.pre_layer_norm.bias',
'backbone.3.post_layer_norm.weight',
'backbone.3.post_layer_norm.bias',
'backbone.4.mlp.fc1.weight',
'backbone.4.mlp.fc1.bias',
'backbone.4.mlp.fc2.weight',
'backbone.4.mlp.fc2.bias',
'backbone.4.attention.W_k.weight',
'backbone.4.attention.W_k.bias',
'backbone.4.attention.W_q.weight',
'backbone.4.attention.W_q.bias',
'backbone.4.attention.W_v.weight',
'backbone.4.attention.W_v.bias',
'backbone.4.attention.W_o.weight',
'backbone.4.attention.W_o.bias',
'backbone.4.pre_layer_norm.weight',
'backbone.4.pre_layer_norm.bias',
'backbone.4.post_layer_norm.weight',
'backbone.4.post_layer_norm.bias',
'backbone.5.mlp.fc1.weight',
'backbone.5.mlp.fc1.bias',
'backbone.5.mlp.fc2.weight',
'backbone.5.mlp.fc2.bias',
'backbone.5.attention.W_k.weight',
'backbone.5.attention.W_k.bias',
'backbone.5.attention.W_q.weight',
'backbone.5.attention.W_q.bias',
'backbone.5.attention.W_v.weight',
'backbone.5.attention.W_v.bias',
'backbone.5.attention.W_o.weight',
'backbone.5.attention.W_o.bias',
'backbone.5.pre_layer_norm.weight',
'backbone.5.pre_layer_norm.bias',
'backbone.5.post_layer_norm.weight',
'backbone.5.post_layer_norm.bias',
'backbone.6.mlp.fc1.weight',
'backbone.6.mlp.fc1.bias',
'backbone.6.mlp.fc2.weight',
'backbone.6.mlp.fc2.bias',
'backbone.6.attention.W_k.weight',
'backbone.6.attention.W_k.bias',
'backbone.6.attention.W_q.weight',
'backbone.6.attention.W_q.bias',
'backbone.6.attention.W_v.weight',
'backbone.6.attention.W_v.bias',
'backbone.6.attention.W_o.weight',
'backbone.6.attention.W_o.bias',
'backbone.6.pre_layer_norm.weight',
'backbone.6.pre_layer_norm.bias',
'backbone.6.post_layer_norm.weight',
'backbone.6.post_layer_norm.bias',
'backbone.7.mlp.fc1.weight',
'backbone.7.mlp.fc1.bias',
'backbone.7.mlp.fc2.weight',
'backbone.7.mlp.fc2.bias',
'backbone.7.attention.W_k.weight',
'backbone.7.attention.W_k.bias',
'backbone.7.attention.W_q.weight',
'backbone.7.attention.W_q.bias',
'backbone.7.attention.W_v.weight',
'backbone.7.attention.W_v.bias',
'backbone.7.attention.W_o.weight',
'backbone.7.attention.W_o.bias',
'backbone.7.pre_layer_norm.weight',
'backbone.7.pre_layer_norm.bias',
'backbone.7.post_layer_norm.weight',
'backbone.7.post_layer_norm.bias',
'backbone.8.mlp.fc1.weight',
'backbone.8.mlp.fc1.bias',
'backbone.8.mlp.fc2.weight',
'backbone.8.mlp.fc2.bias',
'backbone.8.attention.W_k.weight',
'backbone.8.attention.W_k.bias',
'backbone.8.attention.W_q.weight',
'backbone.8.attention.W_q.bias',
'backbone.8.attention.W_v.weight',
'backbone.8.attention.W_v.bias',
'backbone.8.attention.W_o.weight',
'backbone.8.attention.W_o.bias',
'backbone.8.pre_layer_norm.weight',
'backbone.8.pre_layer_norm.bias',
'backbone.8.post_layer_norm.weight',
'backbone.8.post_layer_norm.bias',
'backbone.9.mlp.fc1.weight',
'backbone.9.mlp.fc1.bias',
'backbone.9.mlp.fc2.weight',
'backbone.9.mlp.fc2.bias',
'backbone.9.attention.W_k.weight',
'backbone.9.attention.W_k.bias',
'backbone.9.attention.W_q.weight',
'backbone.9.attention.W_q.bias',
'backbone.9.attention.W_v.weight',
'backbone.9.attention.W_v.bias',
'backbone.9.attention.W_o.weight',
'backbone.9.attention.W_o.bias',
'backbone.9.pre_layer_norm.weight',
'backbone.9.pre_layer_norm.bias',
'backbone.9.post_layer_norm.weight',
'backbone.9.post_layer_norm.bias',
'backbone.10.mlp.fc1.weight',
'backbone.10.mlp.fc1.bias',
'backbone.10.mlp.fc2.weight',
'backbone.10.mlp.fc2.bias',
'backbone.10.attention.W_k.weight',
'backbone.10.attention.W_k.bias',
'backbone.10.attention.W_q.weight',
'backbone.10.attention.W_q.bias',
'backbone.10.attention.W_v.weight',
'backbone.10.attention.W_v.bias',
'backbone.10.attention.W_o.weight',
'backbone.10.attention.W_o.bias',
'backbone.10.pre_layer_norm.weight',
'backbone.10.pre_layer_norm.bias',
'backbone.10.post_layer_norm.weight',
'backbone.10.post_layer_norm.bias',
'backbone.11.mlp.fc1.weight',
'backbone.11.mlp.fc1.bias',
'backbone.11.mlp.fc2.weight',
'backbone.11.mlp.fc2.bias',
'backbone.11.attention.W_k.weight',
'backbone.11.attention.W_k.bias',
'backbone.11.attention.W_q.weight',
'backbone.11.attention.W_q.bias',
'backbone.11.attention.W_v.weight',
'backbone.11.attention.W_v.bias',
'backbone.11.attention.W_o.weight',
'backbone.11.attention.W_o.bias',
'backbone.11.pre_layer_norm.weight',
'backbone.11.pre_layer_norm.bias',
'backbone.11.post_layer_norm.weight',
'backbone.11.post_layer_norm.bias',
'final_layer_norm.weight',
'final_layer_norm.bias',
'lm_head.weight'
"""