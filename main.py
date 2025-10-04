import torch
import torch.nn as nn
from phi_quant import PhiWeightQuantizer
from models import TinyNet
from data import get_mnist_loader

def get_user_choice(prompt, valid_choices):
    while True:
        choice = input(prompt).lower()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please choose from {', '.join(valid_choices)}")

def get_user_integer(prompt, default=None):
    while True:
        choice = input(prompt)
        if not choice:
            return default
        try:
            return int(choice)
        except ValueError:
            print("Invalid input. Please enter an integer.")

def main():
    print("Welcome to the GR-PHI Quantization Tool!")
    print("Building the TinyNet model...")
    net = TinyNet()
    print("Model layers:")
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            print(f"- {name}")

    quant_config = {}

    layers_to_quantize_str = input("Enter the names of the layers to quantize, separated by commas (e.g., fc1,fc2): ")
    layers_to_quantize = [name.strip() for name in layers_to_quantize_str.split(',')]

    for layer_name in layers_to_quantize:
        if not hasattr(net, layer_name):
            print(f"Layer '{layer_name}' not found. Skipping.")
            continue

        print(f"\nConfiguring quantization for layer: {layer_name}")

        per_channel = get_user_choice("Per-channel quantization? (y/n): ", ['y', 'n']) == 'y'
        cluster_size = get_user_integer("Enter cluster size (0 for none, default 0): ", 0)
        e_min = get_user_integer("Enter minimum exponent e_min (optional, press Enter to skip): ", None)
        e_max = get_user_integer("Enter maximum exponent e_max (optional, press Enter to skip): ", None)

        quant_config[layer_name] = {
            "per_channel": per_channel,
            "cluster_size": cluster_size,
            "e_min": e_min,
            "e_max": e_max,
        }

        original_layer = getattr(net, layer_name)
        quantized_layer = PhiWeightQuantizer(
            original_layer,
            **quant_config[layer_name]
        )
        setattr(net, layer_name, quantized_layer)
        print(f"Layer '{layer_name}' wrapped with PhiWeightQuantizer.")

    if not any(isinstance(m, PhiWeightQuantizer) for m in net.modules()):
        print("No layers were quantized. Exiting.")
        return

    print("\nQuantization configuration complete.")

    while True:
        print("\nWhat would you like to do next?")
        print("1. Train the model (QAT)")
        print("2. Export quantized weights")
        print("3. Estimate exponent entropy")
        print("4. Exit")

        choice = get_user_choice("Enter your choice (1-4): ", ['1', '2', '3', '4'])

        if choice == '1':
            print("\n--- Training (QAT) ---")
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            train_loader = get_mnist_loader(train=True)

            steps = get_user_integer("Enter number of training steps (e.g., 100): ", 100)

            print(f"Training for {steps} steps with MNIST data...")

            data_iter = iter(train_loader)
            for step in range(steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                logits = net(x)
                loss = nn.CrossEntropyLoss()(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if (step + 1) % 20 == 0:
                    print(f"Step {step+1}/{steps}, Loss: {loss.item():.4f}")
            print("Training complete.")

        elif choice == '2':
            print("\n--- Exporting Quantized Weights ---")
            net.eval()
            with torch.no_grad():
                for name, module in net.named_modules():
                    if isinstance(module, PhiWeightQuantizer):
                        packed = module.export_packed()
                        print(f"Packed weights for '{name}':")
                        for key, tensor in packed.items():
                            print(f"  - {key}: shape {tensor.shape}, dtype {tensor.dtype}")
                        # Example of saving:
                        # torch.save(packed, f"{name}_packed.pt")
                        # print(f"Saved packed weights to {name}_packed.pt")
            print("Export complete.")

        elif choice == '3':
            print("\n--- Estimating Exponent Entropy ---")
            net.eval()
            with torch.no_grad():
                for name, module in net.named_modules():
                    if isinstance(module, PhiWeightQuantizer):
                        entropy = module.entropy_bits_per_exp()
                        print(f"Estimated exponent entropy for '{name}': {entropy:.4f} bits/symbol")
            print("Entropy estimation complete.")

        elif choice == '4':
            print("Exiting.")
            break

if __name__ == "__main__":
    main()