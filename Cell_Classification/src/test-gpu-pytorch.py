import torch





# Check if CUDA is available
if torch.cuda.is_available():

    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available! Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
    
    # Optionally, you can select a specific GPU by index
    # For example, selecting the first GPU (index 0)
    device = torch.device("cuda:0")
    print(f"\nUsing GPU: {torch.cuda.get_device_name(device)} (index: {torch.cuda.current_device()})")
    
    # Create two tensors and move them to the selected GPU
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    
    # Perform tensor operations on the GPU
    z = x + y
    print("\nx: ", x)
    print("y: ", y)
    print("z: ", z)
    
    # Move the result back to CPU and print
    z_cpu = z.to("cpu")
    print("\nResult moved to CPU: ", z_cpu)
    
    # If you want to utilize multiple GPUs, you can specify the device for each tensor
    if num_gpus > 1:
        print("\nMultiple GPUs detected. Distributing tensors across GPUs.")
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")
        
        x1 = torch.randn(3, 3).to(device1)
        y1 = torch.randn(3, 3).to(device2)
        
        # Perform operations (note: operations across different GPUs require explicit handling)
        z1 = x1.to(device2) + y1  # Move x1 to device2 before addition
        print("\nx1 (GPU 0): ", x1)
        print("y1 (GPU 1): ", y1)
        print("z1 (GPU 1): ", z1)
        
        # Move the result back to CPU
        z1_cpu = z1.to("cpu")
        print("\nResult z1 moved to CPU: ", z1_cpu)
else:
    print("CUDA is not available.")
