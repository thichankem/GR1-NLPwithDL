import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_solution import Transformer
import random
import numpy as np
import torch
import pytest 

from helpers import get_test_modules_from_model, set_seed, TEST_CONFIG

# Fixtures
@pytest.fixture(scope="session")
def seed():
    """Set seed once for the entire test session"""
    set_seed()

@pytest.fixture(scope="module")
def test_model(seed):
    """Create a test model and load snapshot weights"""
    model = Transformer(config=TEST_CONFIG)
    state_dict = torch.load('snapshots/test_model_state_dict.pt')
    model.load_state_dict(state_dict)
    model.eval()
    return model

@pytest.fixture(scope="module")
def test_modules(test_model):
    """Get MLP, attention, and decoder block from model"""
    return get_test_modules_from_model(test_model)

@pytest.fixture
def mlp(test_modules):
    """MLP module for testing"""
    return test_modules[0]

@pytest.fixture
def attention(test_modules):
    """Attention module for testing"""
    return test_modules[1]

@pytest.fixture
def decoder_block(test_modules):
    """Decoder block for testing"""
    return test_modules[2]


# Tests each component
def test_mlp(mlp):
    mlp_input = torch.from_numpy(np.load('snapshots/mlp_input.npy'))
    expected_output = np.load('snapshots/mlp_output.npy')
    
    output = mlp(mlp_input).detach().cpu().numpy()
    
    np.testing.assert_allclose(output, expected_output, atol=1e-5)


def test_attention(attention):
    attention_input = torch.from_numpy(np.load('snapshots/attention_input.npy'))
    expected_output = np.load('snapshots/attention_output.npy')
    
    output = attention(attention_input).detach().cpu().numpy()
    
    np.testing.assert_allclose(output, expected_output, atol=1e-5)


def test_decoder_block(decoder_block):
    decoder_input = torch.from_numpy(np.load('snapshots/decoder_block_input.npy'))
    expected_output = np.load('snapshots/decoder_block_output.npy')
    
    output = decoder_block(decoder_input).detach().cpu().numpy()
    
    np.testing.assert_allclose(output, expected_output, atol=1e-5)


def test_forward(test_model):
    forward_input = torch.from_numpy(np.load('snapshots/forward_input.npy'))
    expected_output = np.load('snapshots/forward_output.npy')
    
    output = test_model(forward_input).detach().cpu().numpy()
    
    np.testing.assert_allclose(output, expected_output, atol=1e-5)


def test_generate(test_model):
    generate_input = torch.from_numpy(np.load('snapshots/generate_input.npy'))
    expected_output = np.load('snapshots/generate_output.npy')
    
    output = test_model.generate(
        x=generate_input,
        num_new_tokens=2,
    ).detach().cpu().numpy()
    
    np.testing.assert_array_equal(output, expected_output)

def test_loss_on_batch(test_model):
    loss_on_batch_input = torch.from_numpy(np.load('snapshots/loss_on_batch_input.npy'))
    expected_output = np.load('snapshots/loss_on_batch_output.npy')
    
    output = test_model.get_loss_on_batch(
        input_ids=loss_on_batch_input
    ).detach().cpu().numpy()
    
    np.testing.assert_allclose(output, expected_output, atol=1e-5)
