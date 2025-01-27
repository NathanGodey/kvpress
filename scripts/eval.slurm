#!/bin/bash
#SBATCH --job-name=eval     # name of job
#SBATCH --output=logs/%x%j.out        # output file (%j = job ID) 
#SBATCH --error=logs/%x%j.out         # error file (%j = job ID)
#SBATCH --time=20:00:00               # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1              # reserving 1 node
#SBATCH --ntasks-per-node=1           # reserving _ tasks (or MPI processes)
#SBATCH --cpus-per-task=10         # reserving 3 CPUs per task (and associated memory)
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread          # deactivating hyperthreading
#SBATCH --account=awr@a100
#SBATCH -C a100
#SBATCH --signal=SIGUSR1@120


module load python/3.11.5

source ../venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/awr/uof65ov/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/awr/uof65ov/hf_cache

export HF_TOKEN=<your_token_here>


srun python evaluation/evaluate.py --dataset ruler --data_dir 4096 \
--model meta-llama/Llama-3.1-70B-Instruct --press_name tova --compression_ratio 0.875 \
--device auto --fraction 0.05

#--dataset loogle --data_dir shortdep_qa
#--dataset ruler --data_dir 4096
