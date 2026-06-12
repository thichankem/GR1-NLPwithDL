"""
Bonus (Q3c): speed up learning within a fixed budget of 100 gradient steps.

Three *different types* of changes, each compounding on the previous one:

  1. Optimization (learning rate): raise lr from 1e-5 to 1e-3.
  2. Architecture (capacity): widen the model from d_model=33 to d_model=132.
  3. Architecture (more attention heads): on top of the wider model, increase the
     number of attention heads from 3 to 6 (d_model 132 -> 264), giving the model
     both more capacity and more attention subspaces.

Trains the baseline and each cumulative configuration, plots all four loss curves
on the same axis, and prints the final loss after 100 steps.
"""
import warnings
warnings.filterwarnings("ignore")

from typing import List

import torch
import matplotlib.pyplot as plt

from model_solution import Transformer, ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_config(name, model_config, learning_rate, batch_size=16, clip=1.0, max_steps=100) -> List[float]:
    cache = f"./datasets/tinystories_10pct_chunk_size_{model_config.context_length}.pt"
    dataset = torch.load(cache)

    torch.manual_seed(0)
    model = Transformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses: List[float] = []
    steps = 0
    for i in range(0, dataset.shape[0], batch_size):
        if steps >= max_steps:
            break
        batch = dataset[i:i + batch_size].to(device)
        optimizer.zero_grad()
        loss = model.get_loss_on_batch(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(loss.item())
        steps += 1

    print(f"{name}: final loss after {len(losses)} steps = {losses[-1]:.4f}", flush=True)
    return losses


if __name__ == "__main__":
    base = ModelConfig(d_model=33, n_heads=3, n_layers=3, context_length=512, vocab_size=50257)
    wide = ModelConfig(d_model=132, n_heads=3, n_layers=3, context_length=512, vocab_size=50257)
    wider = ModelConfig(d_model=264, n_heads=6, n_layers=3, context_length=512, vocab_size=50257)

    runs = {
        "Baseline (lr=1e-5, d=33)": run_config("Baseline", base, 1e-5, clip=1.0),
        "Change 1: +lr=1e-3": run_config("Change 1 (lr)", base, 1e-3, clip=1.0),
        "Change 2: +d_model=132": run_config("Change 2 (lr+width)", wide, 1e-3, clip=1.0),
        "Change 3: +d_model=264, heads=6": run_config("Change 3 (lr+width+heads)", wider, 1e-3, clip=1.0),
    }

    plt.figure(figsize=(8, 5))
    for label, losses in runs.items():
        plt.plot(losses, label=label)
    plt.xlabel("Step"); plt.ylabel("Training loss")
    plt.title("Bonus: compounding speedups over 100 steps")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("bonus_speedup_curves.png"); plt.close()
    print("Saved bonus_speedup_curves.png", flush=True)
