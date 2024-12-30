import os
import logging
from dataclasses import dataclass
import torch
import torch.distributed as dist
from lingua.distributed import (
    setup_torch_distributed, 
    DistributedArgs, 
    get_device_mesh,
    setup_env,
    EnvironmentArgs
)
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

@dataclass
class TestArgs:
    # Minimal version of TrainArgs
    distributed: DistributedArgs = DistributedArgs()
    env: EnvironmentArgs = EnvironmentArgs()
    
def create_dummy_data(batch_size, seq_len, vocab_size):
    # Create random input tokens and labels
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels

def test_distributed(args: TestArgs):
    # Setup environment
    setup_env(args.env)
    setup_torch_distributed(args.distributed)
    
    # Get world info
    world_mesh = get_device_mesh(args.distributed)
    dp_mesh = world_mesh["dp_replicate"]
    dp_degree = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()
    
    if dp_rank == 0:
        logger.info(f"Running with {dist.get_world_size()} processes")
        logger.info(f"Running on dp rank: {dp_rank}")
        logger.info(f"Running on dp size: {dp_degree}")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Model parameters
    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    hidden_size = 768
    
    # Create dummy data
    input_ids, labels = create_dummy_data(batch_size, seq_len, vocab_size)
    input_ids = input_ids.cuda()
    labels = labels.cuda()
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, hidden_size),
        torch.nn.Linear(hidden_size, vocab_size)
    ).cuda()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
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
            
def main():
    """
    Main function that handles config loading similar to train.py
    """
    cli_args = OmegaConf.from_cli()
    
    # Initialize with default config
    default_cfg = OmegaConf.structured(TestArgs())
    
    # Merge with CLI args
    cfg = OmegaConf.merge(default_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    
    test_distributed(cfg)

if __name__ == "__main__":
    main()