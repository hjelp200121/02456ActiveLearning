#!/bin/sh 
### General options 

### -- specify queue -- 
#BSUB -q c02516
#BSUB -gpu "num=1"

### -- set the job Name -- 
#BSUB -J cm_accuracies

### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB

### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/output_%J.out 
#BSUB -e logs/output_%J.err 

source ~/courses/02516/02516env/bin/activate
python3 accuracies.py