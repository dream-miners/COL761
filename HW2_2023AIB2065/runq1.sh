#!bin/bash
module purge
module load compiler/gcc/9.1.0
if [ $# -ne 1 ]; then
	echo "Please specify one argument."
	exit 1
fi
g++ input.cpp -o input
./input $1
read -r graphs<noOfGraphs.txt
chmod ug+x fsg
chmod ug+x gSpan-64
chmod ug+x gaston
for sup in 5 10 25 50 95; do { time ./fsg -s $sup PAFI; } 2>>timeFSG.txt; done
for supFrac in 0.05 0.10 0.25 0.50 0.95; do { time ./gSpan-64 -s $supFrac gSpanInput; } 2>>timeGspan.txt; done
for supAbs in 5 10 25 50 95; do { time ./gaston ((($graphs*$supAbs + 100 -1)/100)) GastonInput; } 2>>timeGaston.txt; done
module purge
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
python3 plotTime.py
module purge