#!/bin/bash

#SBATCH --mem=5000
#SBATCH -J DeepMood

export START_HERE=/nas/student/NicolasKolbenschlag/DeepMood
source $START_HERE/venv/bin/activate

python3 text2friendly.py