import torch
from tqdm import tqdm

from src.data import generate_ising_samples
from src.model import IsingAutoregressiveTransformer
from src.metrics import sampler_score


@torch.no_grad()
def sample_transformer(model, num_samples=64, N=16, device="cuda", temperature=1.0):
    model.eval()

    seq_len = N * N

    tokens = torch.full(
        size=(num_samples, 1),
        fill_value=2,
        device=device,
        dtype=torch.long,
    )

    for _ in tqdm(range(seq_len), desc="autoregressive sampling"):
        logits = model(tokens)
        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    spin_tokens = tokens[:, 1:]
    spins = 2 * spin_tokens - 1

    return spins.reshape(num_samples, N, N).cpu().numpy()


def main():
    N = 16
    num_eval_samples = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        "checkpoints/ising_transformer.pt",
        map_location=device,
    )

    model = IsingAutoregressiveTransformer(
        grid_size=N,
        d_model=128,
        n_heads=4,
        n_layers=4,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Generating held-out true samples...")
    true_samples = generate_ising_samples(
        num_samples=num_eval_samples,
        N=N,
        burn_in=1000,
        thin=10,
    )

    print("Generating model samples...")
    generated_samples = sample_transformer(
        model,
        num_samples=num_eval_samples,
        N=N,
        device=device,
        temperature=1.0,
    )

    score, details = sampler_score(true_samples, generated_samples)

    print("Sampler score:", score)
    print(details)


if __name__ == "__main__":
    main()