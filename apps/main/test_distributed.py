import torch
import torch.distributed as dist
from lingua.distributed import setup_torch_distributed, DistributedArgs, get_device_mesh
import os
import numpy as np

def create_dummy_data(batch_size, seq_len, vocab_size):
    # Create random input tokens and labels
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels

def main():
    # Initialize distributed environment
    dist_args = DistributedArgs(
        dp_replicate=2,  # Number of data parallel replicas
        dp_shard=1,      # Number of FSDP shards
        tp_size=1,       # Tensor parallel size
        fsdp_type="no_shard"  # Start with no sharding for testing
    )
    
    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    setup_torch_distributed(dist_args)
    
    # Get world info
    world_mesh = get_device_mesh(dist_args)
    dp_mesh = world_mesh["dp_replicate"]
    dp_degree = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()
    
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
        
        # Print progress
        if dp_rank == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        
        # Verify distributed is working by all-reducing the loss
        gathered_loss = loss.detach().clone()
        dist.all_reduce(gathered_loss, op=dist.ReduceOp.SUM)
        gathered_loss = gathered_loss / dist.get_world_size()
        
        if dp_rank == 0:
            print(f"Step {step}, Reduced Loss: {gathered_loss.item()}")

if __name__ == "__main__":
    main()