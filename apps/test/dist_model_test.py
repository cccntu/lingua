import os
import logging
import torch
import torch.distributed as dist
from lingua.distributed import (
    setup_torch_distributed, 
    DistributedArgs, 
    get_device_mesh,
    setup_env,
    EnvironmentArgs,
)

logger = logging.getLogger(__name__)

def main():
    # 1. Basic setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Initial setup on rank {local_rank}:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    
    # 2. ALWAYS set device, regardless of device count
    print(f"Setting CUDA device for rank {local_rank}")
    torch.cuda.set_device(local_rank)
    assert torch.cuda.current_device() == local_rank
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # 3. Initialize distributed args
    dist_args = DistributedArgs(
        dp_replicate=world_size,  # Match torchrun's world size
        dp_shard=1,     
        tp_size=1,      
        fsdp_type="no_shard"
    )
    env_args = EnvironmentArgs()
    
    # 4. Setup environment and distributed
    print(f"Setting up distributed environment on rank {local_rank}...")
    setup_env(env_args)
    setup_torch_distributed(dist_args)
    
    # 5. Verify distributed setup
    print(f"\nVerifying setup on rank {local_rank}:")
    print(f"  World size: {dist.get_world_size()}")
    print(f"  Rank: {dist.get_rank()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    
    # 6. Test tensor operations
    x = torch.ones(1, device=f'cuda:{local_rank}')
    dist.all_reduce(x)
    print(f"All reduce result on rank {local_rank}: {x.item()}")
    
    dist.barrier()
    if local_rank == 0:
        print("All processes completed successfully!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()