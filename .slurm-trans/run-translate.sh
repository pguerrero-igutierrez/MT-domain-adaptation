#!/bin/bash
#SBATCH --job-name=translate-ca-eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/pguerrero005/MT/MT-domain-adaptation/.slurm-trans/translate_%j.log
#SBATCH --error=/home/pguerrero005/MT/MT-domain-adaptation/.slurm-trans/translate_%j.err
#SBATCH --chdir=/home/pguerrero005/MT/MT-domain-adaptation

source /home/pguerrero005/envs/MTproject_3.11/bin/activate

export HF_HOME="/home/pguerrero005/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/pguerrero005/.cache/huggingface"
export HF_HUB_CACHE="/home/pguerrero005/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"

python scripts/05_translate-ca-literary.py --langs eu ca --resume --batch-size 32 --max-tokens 512

echo "Job finished at $(date)"
