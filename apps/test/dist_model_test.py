import os
import logging
import torch
import torch.distributed as dist
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
)

logger = logging.getLogger(__name__)

def main():
    # 1. Basic CUDA setup and info
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Initial setup:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    print(f"  Local rank: {local_rank}")
    print(f"  World size: {world_size}")
    
    # 2. Set device first
    torch.cuda.set_device(local_rank)
    print(f"Set CUDA device to: {torch.cuda.current_device()}")
    
    # 3. Initialize process group directly first
    print("\nInitializing process group directly...")
    dist.init_process_group(backend="nccl")
    print("Direct process group initialization successful!")
    
    # 4. Create and move a tensor to verify CUDA access
    try:
        x = torch.ones(1, device=f'cuda:{local_rank}')
        print(f"Successfully created tensor on {x.device}")
        
        # Test all_reduce
        dist.all_reduce(x)
        print(f"Successfully completed all_reduce, result: {x.item()}")
    except Exception as e:
        print(f"Error in tensor operations: {e}")
        raise
    
    # 5. Now try lingua setup
    print("\nSetting up lingua distributed args...")
    dist_args = DistributedArgs(
        dp_replicate=world_size,
        dp_shard=1,
        tp_size=1,
        fsdp_type="no_shard"
    )
    
    print(f"Proceeding with process cleanup...")
    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {local_rank} completed successfully!")

if __name__ == "__main__":
    main()