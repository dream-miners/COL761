#!/bin/bash 
module purge
module load compiler/gcc/9.1.0
chmod ug+x generateDataset_d_dim_hpc_compiled
./generateDataset_d_dim_hpc_compiled $1 $2

module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load pythonpackages/3.6.0/scikit-learn/0.21.2/gnu
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <dimension> <plot.png>"
    exit 1
fi
dataset="$1_generated_dataset_$2D.dat"
dimension="$2"
plot_name="$3"
python script.py "$dataset" "$dimension" "$plot_name"
module purge
