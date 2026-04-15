import os
import torch
import torch.nn as nn
import torch.optim as optim

# Allow CPU fallback for unimplemented MPS ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device():
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )


class TinyMLP(nn.Module):
    def __init__(self, d_in=32, d_hidden=64, d_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


def main():
    dev = device()
    print(f"Using device: {dev}")

    torch.manual_seed(0)
    # Synthetic dataset: 4096 samples, 32-dim features, 10 classes
    x = torch.randn(4096, 32, device=dev)
    y = torch.randint(0, 10, (4096,), device=dev)

    model = TinyMLP().to(dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        print(f"epoch {epoch + 1}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
