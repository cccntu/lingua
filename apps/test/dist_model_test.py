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
    parallelize_model
)

logger = logging.getLogger(__name__)

def create_dummy_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels

def main():
    # Get local rank from environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Fixed distributed args
    dist_args = DistributedArgs(
        dp_replicate=2,  # Number of data parallel replicas 
        dp_shard=1,      # No FSDP sharding
        tp_size=1,       # No tensor parallelism
        fsdp_type="no_shard"
    )
    env_args = EnvironmentArgs()
    
    # Set the device first
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"Local rank {local_rank} >= GPU count {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    
    # Setup distributed environment
    setup_env(env_args)
    setup_torch_distributed(dist_args)
    
    # Get world info
    world_mesh = get_device_mesh(dist_args)
    dp_mesh = world_mesh["dp_replicate"]
    dp_degree = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()
    
    if dp_rank == 0:
        logger.info(f"Running with {dist.get_world_size()} processes")
        logger.info(f"Running on dp rank: {dp_rank}")
        logger.info(f"Running on dp size: {dp_degree}")
    
    # Set random seed
    torch.manual_seed(42 + local_rank)  # Different seed per rank
    
    # Model parameters
    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    hidden_size = 768
    
    # Create dummy data
    input_ids, labels = create_dummy_data(batch_size, seq_len, vocab_size)
    input_ids = input_ids.cuda()
    labels = labels.cuda()
    
    # Create model with meta device
    with torch.device('meta'):
        model = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, hidden_size),
            torch.nn.Linear(hidden_size, vocab_size)
        )
    
    # Parallelize the model
    logger.info("Parallelizing model...")
    model = parallelize_model(
        model,
        world_mesh,
        None,  # No model args needed for this simple test
        dist_args,
    )
    
    # Initialize on GPU
    model = model.to_empty(device="cuda")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)  # Same init across ranks
        model[0].weight.copy_(torch.randn_like(model[0].weight))
        model[1].weight.copy_(torch.randn_like(model[1].weight))
        model[1].bias.copy_(torch.randn_like(model[1].bias))
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for step in range(5):
        # Forward pass
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print progress from rank 0
        if dp_rank == 0:
            logger.info(f"Step {step}, Loss: {loss.item()}")
        
        # Verify distributed is working by all-reducing the loss
        gathered_loss = loss.detach().clone()
        dist.all_reduce(gathered_loss, op=dist.ReduceOp.SUM)
        gathered_loss = gathered_loss / dist.get_world_size()
        
        if dp_rank == 0:
            logger.info(f"Step {step}, Reduced Loss: {gathered_loss.item()}")

if __name__ == "__main__":
    main()