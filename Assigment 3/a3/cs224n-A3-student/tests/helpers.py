import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple
import random
import numpy as np
import torch

from model_solution import ModelConfig
from model_solution import Transformer, MLP, CausalAttention, DecoderBlock


TEST_CONFIG = ModelConfig(
    d_model=48,
    n_heads=2,
    n_layers=2,
    context_length=16,
    vocab_size=16,
)

def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_test_modules_from_model(model: Transformer) -> Tuple[MLP, CausalAttention, DecoderBlock]:

    decoder_block: DecoderBlock = model.backbone[0]
    mlp: MLP = decoder_block.mlp
    attention: CausalAttention = decoder_block.attention

    return mlp, attention, decoder_block
