# Transformer Classifier for Reweighting

## Using shifter on Perlmutter
All the libraries required to run the code in the repo can be acessed through the docker image ```vmikuni/tensorflow:ngc-23.12-tf2-v1```. You can test it locally by doing:
```bash
shifter --image=vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18
```

Alternatively, you can use the tensorflow module provided by NERSC with

```bash
module load tensorflow
```

You can run the training code using a single GPU after loading the module/docker container using the command

```bash
cd scripts
python train.py
```

You can also run the code using multiple GPUs by first starting an interactive job with the commands:

```bash
salloc -C gpu -q interactive  -t 30 -n 4 --ntasks-per-node=4 --gpus-per-task=1  -A m4045 --gpu-bind=none  --image vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18
```

in case you want to use the docker container or

```bash
salloc -C gpu -q interactive  -t 30 -n 4 --ntasks-per-node=4 --gpus-per-task=1  -A m4045 --gpu-bind=none
```
 using the module.

There you can run the code with the multiple GPUs using the command:

```bash
srun --mpi=pmi2 shifter python train.py
```
with the docker container or simply
```bash
module load tensorflow
srun python train.py
```
with the module.