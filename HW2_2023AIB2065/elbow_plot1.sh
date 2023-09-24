#!/bin/bash 
module purge
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load pythonpackages/3.6.0/scikit-learn/0.21.2/gnu
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <dimension> <plot.png>"
    exit 1
fi
dataset="$1"
dimension="$2"
plot_name="$3"
python script.py "$dataset" "$dimension" "$plot_name"
module purge
