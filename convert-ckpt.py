#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# Set required env vars for distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

from lingua.transformer import BaseTransformer, BaseTransformerArgs
from lingua.checkpoint import load_from_checkpoint

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    save_safetensors = None


def export_checkpoint(ckpt_path: Path, model, save_format: str) -> Path:
    """
    Loads the given checkpoint path, then saves the model state
    in the requested format (pt or safetensors).
    Returns the output path.
    """
    load_from_checkpoint(ckpt_path, model, model_key="model")

    out_name = f"{ckpt_path.name}_py_state_dict.{save_format}"
    out_path = ckpt_path.parent / out_name

    if save_format == "pt":
        torch.save(model.state_dict(), out_path)
    else:
        if not save_safetensors:
            raise RuntimeError("safetensors is not installed; install it or use --save-format pt")
        save_safetensors(model.state_dict(), str(out_path))

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="Path to the directory containing config.yaml and checkpoints/ subdir.")
    parser.add_argument("--step", type=int,
                        help="Step number of a single checkpoint to export. If not provided, exports all checkpoints.")
    parser.add_argument("--save-format", default="pt", choices=["pt", "safetensors"],
                        help="Output format. Options: pt, safetensors.")
    parser.add_argument("--dist-backend", default="nccl", help="Distributed backend.")
    parser.add_argument("--rank", type=int, default=0, help="Process rank.")
    parser.add_argument("--world-size", type=int, default=1, help="World size.")
    args = parser.parse_args()

    # Initialize distributed environment
    dist.init_process_group(backend=args.dist_backend, rank=args.rank, world_size=args.world_size)

    # Load config
    config_path = args.dump_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = OmegaConf.load(config_path)
    if not hasattr(config, "model"):
        raise ValueError("No 'model' section in config.")

    # Filter out extra fields that BaseTransformerArgs doesn't expect
    model_args_dict = dict(config.model)
    valid_keys = set(BaseTransformerArgs.__dataclass_fields__.keys())
    filtered_args = {k: v for k, v in model_args_dict.items() if k in valid_keys}
    extra_keys = set(model_args_dict.keys()) - valid_keys
    if extra_keys:
        print(f"Warning: discarding unrecognized parameters in config.model: {extra_keys}")

    # Create model
    model_args = BaseTransformerArgs(**filtered_args)
    model = BaseTransformer(model_args).cuda()

    # Checkpoints directory
    ckpt_dir = args.dump_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found at: {ckpt_dir}")

    # Export single or all checkpoints
    if args.step is not None:
        # Single
        ckpt_path = ckpt_dir / f"{args.step:010d}"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        out_path = export_checkpoint(ckpt_path, model, args.save_format)
        print(f"Exported: {out_path}")
    else:
        # All
        exported = []
        for ckpt_path in sorted(ckpt_dir.iterdir()):
            # Check if directory name is numeric (typical step naming convention)
            if ckpt_path.is_dir() and ckpt_path.name.isdigit():
                out_path = export_checkpoint(ckpt_path, model, args.save_format)
                exported.append(out_path)
        if not exported:
            raise FileNotFoundError(f"No numeric-named checkpoint directories found in {ckpt_dir}")

        print("Exported checkpoints:")
        for p in exported:
            print(p)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
