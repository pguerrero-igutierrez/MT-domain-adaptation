#!/bin/bash
#SBATCH --job-name=finetune-general
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/pguerrero005/MT/MT-domain-adaptation/.slurm-finetune/finetune_cav1_%j.log
#SBATCH --error=/home/pguerrero005/MT/MT-domain-adaptation/.slurm-finetune/finetune_cav1_%j.err
#SBATCH --chdir=/home/pguerrero005/MT/MT-domain-adaptation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pguerrero005@ikasle.ehu.eus


source /home/pguerrero005/envs/MTproject_3.11/bin/activate

export HF_HOME="/home/pguerrero005/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/pguerrero005/.cache/huggingface"
export HF_HUB_CACHE="/home/pguerrero005/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/09_finetuning_literaryv1.py

echo "Job finished at $(date)"