#!/bin/bash
#SBATCH --job-name=weight-classifier-optimization
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output_weight_opt_2024-11-05.txt
#SBATCH --error=job_error_weight_opt_2024-11-05.txt

# carregar versão python
module load Python/3.9

# ativar ambiente
source ./env/bin/activate

# executar .py
python3 src/training/optimizer.py