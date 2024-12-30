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
    get_world_size
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
    
    # Set device first
    torch.cuda.set_device(local_rank)
    print(f"Set CUDA device to: {torch.cuda.current_device()}")

    # 2. Setup distributed environment
    # Calculate dp_replicate based on world size
    dp_replicate = world_size  # This ensures we match torchrun's world size
    
    dist_args = DistributedArgs(
        dp_replicate=dp_replicate,  # Match torchrun's world size
        dp_shard=1,     # No FSDP sharding
        tp_size=1,      # No tensor parallelism
        fsdp_type="no_shard"
    )
    env_args = EnvironmentArgs()
    
    print(f"\nDistributed Args:")
    print(f"  dp_replicate: {dist_args.dp_replicate}")
    print(f"  dp_shard: {dist_args.dp_shard}")
    print(f"  tp_size: {dist_args.tp_size}")
    print(f"  Calculated world size: {dist_args.dp_replicate * dist_args.dp_shard * dist_args.tp_size}")
    
    # Setup distributed
    print("\nSetting up distributed environment...")
    setup_env(env_args)
    setup_torch_distributed(dist_args)
    
    actual_world_size = get_world_size()
    print(f"\nAfter setup:")
    print(f"  World size: {actual_world_size}")
    print(f"  Current device: {torch.cuda.current_device()}")
    
    print(f"\nRank {local_rank} completed successfully!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()