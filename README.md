DDP scipt can be executed with `python main_DDP.py` in terminal/slurm script.

MP script should be executed with `torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 main_MP.py`, where `nproc_per_node`and `nnodes` should be changed accordingly.
