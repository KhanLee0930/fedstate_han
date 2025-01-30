# fedstate

```
cp /app1/common/singularity-img/hopper/pytorch/pytorch_2.1.0_cuda_12.1_ngc_23.07.sif /home/svu/e1143336/container/fedstate2.sif
module load singularity
singularity exec -e /home/svu/e1143336/container/fedstate.sif bash
pip install --prefix=/home/svu/e1143336/hopper_pypkg/fedstate.sif/ numpy
export PYTHONPATH=$PYTHONPATH:/home/svu/e1143336/hopper_pypkg/fedstate.sif/local/lib/python3.10/dist-packages
```

qsub main.sh
qstat -xfn


## 

## Interactive Mode
qsub -I -l select=1:ngpus=1 -l walltime=2:00:00 -P H100011 -q normal
module load singularity
singularity exec /home/svu/e1143336/container/fedsate_latest.sif bash
 