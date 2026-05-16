#!/bin/bash
#SBATCH --job-name=finetune-cav1_tokenmatched
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/MT/MT-domain-adaptation/.slurm-finetune/finetune_cav1_tokenmatched_%j.log
#SBATCH --error=/home/igutierrez134/MT/MT-domain-adaptation/.slurm-finetune/finetune_cav1_tokenmatched_%j.err
#SBATCH --chdir=/home/igutierrez134/MT/MT-domain-adaptation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus


source /home/igutierrez134/envs/MTproject_3.11/bin/activate

export HF_TOKEN=HF_TOKEN
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/09_finetuning_literaryv1_tokenmatched.py

echo "Job finished at $(date)"