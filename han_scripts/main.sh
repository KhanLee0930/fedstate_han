#!/bin/bash
#PBS -P H100011
#PBS -j oe
#PBS -k oed
#PBS -N pytorch
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
ROOT=/home/svu/e1143336/fedstate/han_scripts
image="/home/svu/e1143336/container/fedstate.sif"

module load singularity
singularity exec -e $image bash << EOF > $ROOT/log/stdout.$PBS_JOBID 2> $ROOT/log/stderr.$PBS_JOBID

export PYTHONPATH=$PYTHONPATH:/home/svu/e1143336/hopper_pypkg/fedstate.sif/local/lib/python3.10/dist-packages
python /home/svu/e1143336/fedstate/han_scripts/main.py
nvidia-smi
echo "Hello World"
EOF
