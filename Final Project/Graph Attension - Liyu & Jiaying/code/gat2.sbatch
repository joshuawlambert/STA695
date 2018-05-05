#!/bin/bash
#SBATCH --mem=16G -p pgpu --cpus-per-task=1 --gres=gpu:1 --time=40:55:00
#SBATCH --mail-user=liyu.gong@uky.edu
#SBATCH --mail-type=ALL

source activate gpu
for i in `seq 1 100`; do
    python gat2.py
done
