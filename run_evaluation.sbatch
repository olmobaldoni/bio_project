#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G # Inserisci la quantità di memoria richiesta
#SBATCH --gres=gpu:1 # Modifica con il numero di GPU richieste
#SBATCH --time=24:00:00
#SBATCH --account=ai4bio2023
#SBATCH --job-name=textual_inversion
#SBATCH --partition=all_usr_prod
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --constraint="gpu_2080Ti_11G" 

# Attiva l'ambiente conda in cui hai installato PyTorch
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3' 'shell.bash' 'hook' 2> /dev/null)"
# $? è il codice di uscita dell'ultimo comando eseguito, 0 se è andato a buon fine
if [ $? -eq 0 ]; then
    eval "$__conda_setup" # Esegue il comando restituito da conda init
else # Se conda non è installato, modifica il PATH con il percorso di Anaconda
    if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/bin/:$PATH"
    fi 
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate bio_env
export PYTHONPATH='/usr/local/anaconda3/bin/python' # Modifica percorso Python se necessario

python /homes/obaldoni/bio_project/textual_inversion/eval.py