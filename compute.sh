#!/bin/bash
# FILENAME:  vae2.sh

#SBATCH -J resnet
#SBATCH -o /anvil/projects/x-mcb090163/saptorshig/logs/test_fenics.txt
#SBATCH -e /anvil/projects/x-mcb090163/saptorshig/logs/test_fenics.txt
#SBATCH -p gpu
#SBATCH -A mcb090163-gpu
#SBATCH --nodes=1
##SBATCH -n 5
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=2-00:00:00

 
# export NCCL_DEBUG=INFO

echo "NODELIST="${SLURM_NODELIST}


#source ~/.bashrc

echo $TMP
echo $TMPDIR
echo $TMP


source ~/miniconda3/bin/activate

echo "activating fenics environment"

conda activate fenicsproject
mpirun -n 2 python3 server.py > logs/fenics_output_$SLURM_NODELIST.txt 2> logs/fenics_error_$SLURM_NODELIST.txt &
conda deactivate

sleep 40

echo "activating rl environment"

conda activate rl
python3 resnet_50.py > logs/ray_output_$SLURM_NODELIST.txt 2> logs/ray_error_$SLURM_NODELIST.txt


