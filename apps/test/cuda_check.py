import torch
import sys

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {props.name}")
            print(f"  Total memory: {props.total_memory / 1024**2:.0f}MB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("No CUDA devices available!")
        sys.exit(1)

    # Try to create a simple tensor on each GPU
    print("\nTesting GPU tensor creation:")
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            x = torch.ones(1).cuda()
            print(f"Successfully created tensor on GPU {i}")
            print(f"Tensor device: {x.device}")
        except Exception as e:
            print(f"Failed to create tensor on GPU {i}: {e}")

if __name__ == "__main__":
    main()