#!/bin/bash
# FILENAME:  vae2.sh

#SBATCH -J resnet
#SBATCH -o logs/test_fenics.txt
#SBATCH -e logs/test_fenics.txt
#SBATCH -p gpu
#SBATCH -A mcb090163-gpu
#SBATCH --nodes=1
##SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
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

if [ ! -d "logs" ]; then
    mkdir logs
fi

echo "activating fenics environment"

conda activate fenicsproject
mpirun -n 2 python3 server_fenics_pickle.py --port $1 > logs/fenics_output_$SLURM_NODELIST.txt 2> logs/fenics_error_$SLURM_NODELIST.txt &
conda deactivate

sleep 40

echo "activating rl environment"

conda activate rl
python3 clip_action.py --port $1 > logs/ray_output_$SLURM_NODELIST.txt 2> logs/ray_error_$SLURM_NODELIST.txt


