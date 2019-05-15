#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=resnet_%j.out

module purge
module load python3/intel/3.6.3
source ~/pytorch-cpu/py3.6.3/bin/activate


python -c "print('start main.py')"
python main.py --cuda --model vgg --save vgg.pt --epochs 10 --pretrained True
python -c "print('end of epochs')"
