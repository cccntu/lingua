import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def main():
    # 1. Print CUDA info first
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # 2. Get local rank and set device before anything else
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"Setting device for rank {local_rank}")
    
    # Set the device first
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"Local rank {local_rank} >= GPU count {torch.cuda.device_count()}")
        
    # 3. Try each CUDA operation separately
    try:
        torch.cuda.set_device(local_rank)
        print(f"Successfully set CUDA device to {local_rank}")
        
        # Test device is working
        x = torch.zeros(1, device=f'cuda:{local_rank}')
        print(f"Successfully created tensor on device {x.device}")
        
        # Initialize process group
        print("Initializing process group...")
        dist.init_process_group(backend="nccl")
        print(f"Process group initialized successfully!")
        print(f"World size: {dist.get_world_size()}")
        print(f"Rank: {dist.get_rank()}")
        
        # Test basic distributed operation
        test_tensor = torch.ones(1, device=f'cuda:{local_rank}')
        dist.all_reduce(test_tensor)
        print(f"Successfully completed all_reduce, result: {test_tensor.item()}")
        
        dist.barrier()
        print("Successfully created barrier")
        
        dist.destroy_process_group()
        print("Successfully destroyed process group")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()