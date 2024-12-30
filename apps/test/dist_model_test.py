import os
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

def create_dummy_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels

def main():
    # 1. Basic CUDA and device setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"Setting device for rank {local_rank}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # 2. Initialize process group
    print("Initializing process group...")
    dist.init_process_group(backend="nccl")
    print(f"Process group initialized! Rank {dist.get_rank()} of {dist.get_world_size()}")
    
    # 3. Model parameters
    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    hidden_size = 768
    
    # 4. Create model
    print("Creating model...")
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, hidden_size),
        torch.nn.Linear(hidden_size, vocab_size)
    ).to(device)
    
    # 5. Wrap model with DDP
    print("Wrapping model with DDP...")
    model = DDP(model, device_ids=[local_rank])
    print("Model created and wrapped!")
    
    # 6. Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 7. Training loop
    print("Starting training loop...")
    model.train()
    for step in range(3):
        # Create dummy data
        input_ids, labels = create_dummy_data(batch_size, seq_len, vocab_size)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress
        if dist.get_rank() == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        dist.barrier()
    
    print(f"Rank {dist.get_rank()} completed training!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()