#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --partition=ma
#SBATCH --account=ma
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sn634/NeuralSteganography/slurm_robustness_%j.out

eval "$(conda shell.bash hook)"
conda activate emu3_env
cd /home/sn634/NeuralSteganography
python scripts/run_robustness_eval.py
