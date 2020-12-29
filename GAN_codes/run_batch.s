#!/bin/bash -e

# requesting resources and other parameters
# submit job with `sbatch run_batch.s`
# monitor running jobs using `squeue`, cancel with `scancel`, etc

#SBATCH --job-name=
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4,p40_4,v100_sxm2_4,v100_pci_2
  # these partitions have better GPUs 
#SBATCH --mem=100GB
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --output=output.txt
#SBATCH --error=error.txt

#PRINCE PRINCE_GPU_COMPUTE_MODE=DEFAULT

# loading relevant environments and packages-------------------------------------------------
module purge
module load anaconda3/5.3.1
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
conda activate /scratch/cc6580/carlos_research/nanop_env
#export DISPLAY=""

# for working without X-winndow connection in batch mode
# then do not need agg code in script (convenient)
export MPLBACKEND="agg"

conda info --envs
conda list | grep cudnn

module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32

# run your python script --------------------------------------------------------------------
# the 1st argument is the start folder
# the 2nd is the end folder for training images
# the 3rd is 0 if you want to start a fresh training, 1 if you want to continue where you left off
# the 4th is the size/length of `z` latent vector (i.e. size of generator input)
# the 5th is the start epoch number from which you'd like to start the continued training, models will be loaded from one prior	epoch
# the 6th is the learning rate for the G optimizer
# the 7th is the learning rate for the D optimizer
# the 8th is the path to the saved logs file, which contains saved lists: G_losses and D_losses
# note that in order to continue training, you must have img_list.ts and last_imgs.ts files present in the folder

python -b nanop_GAN.py 1 3 0 100 501 0.0001 0.0004 logs.json
