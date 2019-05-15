#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=de_%j.out

module purge
module load python3/intel/3.6.3
source ~/pytorch-cpu/py3.6.3/bin/activate


python -c "print('start de.py')"
python de.py
python -c "print('end')"