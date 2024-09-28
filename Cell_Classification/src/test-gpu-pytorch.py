

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    
    # Get the current GPU device
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_index = torch.cuda.current_device()

    print(f"Using GPU: {gpu_name} (index: {gpu_index})")

    # Create two tensors and move them to the GPU
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)

    # Perform tensor operations on the GPU
    z = x + y
    print("x: ", x)
    print("y: ", y)
    print("z: ", z)

    # Move the result back to CPU and print
    z_cpu = z.to("cpu")
    print("Result moved to CPU: ", z_cpu)
else:
    print("CUDA is not available.")
