import torch

def gpu_test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {gpu_name}")

        # Perform a simple computation on the GPU
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        print("GPU computation successful.")
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    gpu_test()

