#!/bin/sh
#SBATCH -A uppmax2025-3-5
#SBATCH -p node
#SBATCH -M snowy
#SBATCH --gres=gpu:1   
#SBATCH --time=20:0:00 
#SBATCH -J mt_proj 
#SBATCH --mail-type=END
#SBATCH --mail-user=lanwireless@outlook.com

module load python3/3.10.8
source /domus/h1/yifanh/.venv/bin/activate
python cn2en.py
