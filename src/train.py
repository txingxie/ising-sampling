import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.data import generate_ising_samples
from src.model import IsingAutoregressiveTransformer, spins_to_tokens


def train_step(model, batch_spins, optimizer, device):
    model.train()

    batch_spins = batch_spins.to(device)
    tokens = spins_to_tokens(batch_spins)

    B, L = tokens.shape

    bos = torch.full(
        size=(B, 1),
        fill_value=2,
        device=device,
        dtype=torch.long,
    )

    input_tokens = torch.cat([bos, tokens[:, :-1]], dim=1)
    target_tokens = tokens

    logits = model(input_tokens)

    loss = F.cross_entropy(
        logits.reshape(-1, 2),
        target_tokens.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    N = 16
    num_train_samples = 20000
    batch_size = 128
    epochs = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Generating training samples...")
    train_samples = generate_ising_samples(
        num_samples=num_train_samples,
        N=N,
        burn_in=1000,
        thin=10,
    )

    train_tensor = torch.tensor(train_samples, dtype=torch.long)
    dataset = TensorDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = IsingAutoregressiveTransformer(
        grid_size=N,
        d_model=128,
        n_heads=4,
        n_layers=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        total_loss = 0.0

        for (batch_spins,) in tqdm(loader, desc=f"epoch {epoch + 1}"):
            loss = train_step(model, batch_spins, optimizer, device)
            total_loss += loss * batch_spins.shape[0]

        avg_loss = total_loss / len(dataset)
        print(f"epoch {epoch + 1}: loss = {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "grid_size": N,
        },
        "checkpoints/ising_transformer.pt",
    )

    print("Saved checkpoint to checkpoints/ising_transformer.pt")


if __name__ == "__main__":
    main()