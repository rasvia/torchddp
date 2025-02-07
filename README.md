Some instructions:

- DDP scipt can be executed with `python main_DDP.py` in terminal/slurm script.

- MP script should be executed with `torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 main_MP.py`, where `nproc_per_node`and `nnodes` should be changed accordingly.
  - `n_proc_per_node` should be `total_GPU_num // GPU_num_for_MP`; i.e, if there are total 8 GPUs available, and 2 GPUs are used to distribute model, then `n_proc_per_node` should be 4.
