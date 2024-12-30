import os
import sys
import torch
import torch.distributed as dist

def main():
    # 1. Check CUDA first
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # 2. Print environment variables
    print("\nEnvironment variables:")
    for var in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Get local rank from environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"\nLocal rank: {local_rank}")
    
    # 3. Try to set the device
    if torch.cuda.is_available():
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            print(f"Set cuda device to: {torch.cuda.current_device()}")
        else:
            print(f"Error: local_rank {local_rank} >= number of GPUs {torch.cuda.device_count()}")
            sys.exit(1)
    
    # 4. Try initializing process group
    try:
        print("\nInitializing process group...")
        dist.init_process_group(backend="nccl")
        print("Process group initialized!")
        
        print(f"\nPost-initialization:")
        print(f"World size: {dist.get_world_size()}")
        print(f"Rank: {dist.get_rank()}")
        
        # 5. Test simple tensor allreduce
        print("\nTesting allreduce...")
        tensor = torch.ones(1).cuda()
        dist.all_reduce(tensor)
        print(f"Allreduce result: {tensor.item()}")
        
        dist.destroy_process_group()
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        raise

if __name__ == "__main__":
    main()