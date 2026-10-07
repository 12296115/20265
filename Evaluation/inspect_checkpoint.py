import torch

MODEL_PATH = "./deberta_v3_large_phishing_model/pytorch_model.bin"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("=" * 70)
print("EXACT CUSTOM MODEL WEIGHTS")
print("=" * 70)

for name, tensor in checkpoint.items():
    if not any(x in name for x in [
        "feature_projection",
        "context_projection",
        "gate",
        "fusion",
        "classifier"
    ]):
        continue

    print(f"\n{name}")
    print("Shape:", tuple(tensor.shape))

    # Print actual values for the small custom layers only
    if tensor.numel() <= 1000:
        print(tensor)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)