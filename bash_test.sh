#!/bin/bash

# for abide
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config configs/params/abide_tiny.yaml;
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config configs/params/abide_small.yaml;
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config configs/params/abide_medium.yaml;
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config configs/params/abide_large.yaml;
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config configs/params/abide_huge.yaml;