"""A simple training loop for our transformer model"""
import warnings
warnings.filterwarnings("ignore")
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Optional
from jaxtyping import Int
from torch import Tensor
import torch
import matplotlib.pyplot as plt

from model_solution import Transformer, ModelConfig


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using Mac MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


def get_chunked_tinystories(
    chunk_size: int,
) -> Int[Tensor, "num_chunks chunk_size"]:

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load tiny stories dataset
    train_dataset = load_dataset("roneneldan/TinyStories")["train"]

    # We'll just grab the first 1%
    train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.01)))

    # Tokenize the dataset
    chunks: List[List[int]] = []
    current_chunk: List[int] = []
    for row in tqdm(train_dataset, desc="Tokenizing dataset"):
        document: str = row["text"]
        tokens: List[int] = tokenizer(document, truncation=True, max_length=chunk_size).input_ids

        # Fill current chunk up to chunk_size
        current_chunk.extend(tokens)
        if len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size])
            # Reset the current chunk
            current_chunk = current_chunk[chunk_size:]

    # Sanity checks
    assert all(len(chunk) == chunk_size for chunk in chunks)

    return torch.tensor(chunks, dtype=torch.long)


def plot_results(
    losses: List[float],
    grad_norms: List[float],
    save_path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel - Loss curve
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Right panel - Gradient norm
    ax2.plot(grad_norms)
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Grad Norm')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(
    learning_rate: float,
    gradient_clipping: Optional[float],
    model_config: ModelConfig,
    batch_size: int,
    max_steps: Optional[int] = None,
) -> None:

    if gradient_clipping is None:
        # This lets us just get the grad norm but we don't clip
        gradient_clipping = float("inf")

    chunk_size: int = model_config.context_length
    cached_dataset_path: str = f"./datasets/tinystories_10pct_chunk_size_{chunk_size}.pt"
    os.makedirs(os.path.dirname(cached_dataset_path), exist_ok=True)
    
    if os.path.exists(cached_dataset_path):
        dataset = torch.load(cached_dataset_path)
    else:
        dataset: Int[Tensor, "num_chunks chunk_size"] = get_chunked_tinystories(chunk_size)
        torch.save(dataset, cached_dataset_path)


    # Create dense batches of [batch_size, seq_len]
    model = Transformer(model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_chunks: int = dataset.shape[0]

    losses: List[float] = []
    grad_norms: List[float] = []
    num_steps_completed: int = 0

    if max_steps is not None:
        tqdm_max_steps = min(max_steps, num_chunks // batch_size)
    else:
        tqdm_max_steps = num_chunks // batch_size

    for i in tqdm(range(0, num_chunks, batch_size), desc="Training", total=tqdm_max_steps):

        if max_steps is not None and num_steps_completed >= max_steps:
            break

        if num_steps_completed % 10 == 0 and num_steps_completed > 0:
            plot_results(losses, grad_norms, save_path=f"./losses_and_grad_norms.png")

        batch: Int[Tensor, "batch_size chunk_size"] = dataset[i:i+batch_size].to(device)

        optimizer.zero_grad()

        # Forward pass
        loss = model.get_loss_on_batch(batch)

        # Backward pass
        loss.backward()

        # Clip gradients
        with torch.no_grad():
            grad_norm: float = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping).item()
            grad_norms.append(grad_norm)

        optimizer.step()

        losses.append(loss.item())

        num_steps_completed += 1


    # Done with training, plot results in single plot
    plot_results(losses, grad_norms, save_path="./losses_and_grad_norms.png")



if __name__ == "__main__":

    tiny_model_config = ModelConfig(
        d_model=33,
        n_heads=3,
        n_layers=3,
        context_length=512,
        vocab_size=50257,
    )

    train(
        learning_rate=1e-5,
        gradient_clipping=1,
        model_config = tiny_model_config,
        batch_size=16,
        max_steps=100,
    )

