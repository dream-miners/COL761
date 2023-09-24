i. Files included
'input.cpp'
'gaston'
'gSpan-64'
'elbow_plot1.sh'
'elbow_plot2.sh'
'runq1.sh'
'fsg'
'gSpan-64'
'generateDataset_d_dim_hpc_compiled'
'plotTime.py'
'script.py'


ii. Instructions to execute the code and files required by them

1. For question 1, 'runq1.sh' requires one argument which is the dataset for which we need to mine the subpragohs for. For example, it would be run as follows:

sh runq1.sh 167.txt_graph

This bash script uses 'input.cpp' a program we created to convert the format of the input file to suitable formats in order to run FSG, gSpan and Gaston algorithm on them. It creates 4 files - 'PAFI', 'gSpanInput', 'GastonInput' and 'noOfGraphs.txt'. 'noOfGraphs.txt' is a one line text file which contains the total number of graphs in the original file.

Bash commands are used to run all three algorithms and store their running time in files named 'timeFSG.txt', 'timeGspan.txt' and 'timeGaston.txt'. These files are then used by a python file 'plotTime.py' to create the graph of running time of all three algorithms.

2. For question 2, since it was not clear in the question or via the discussion on Piazza what were the input parameters, we have created two script files - 'elbow_plot1.sh' and 'elbow_plot2.sh'.

'elbow_plot1.sh' requires 3 arguments in this order - (a) the path to data set, (b) the dimension of data set, and (c) the name and/or the path under which the plot is to be saved. For example,

sh elbow_plot1.sh /home/baadalvm/AIB232065_generated_dataset_4D.dat 4 q3_4_2023AIB2065.png

'elbow_plot1.sh' uses the python file 'script.py' to create the elbow plot for the given data set.


'elbow_plot.sh' requires 3 arguments as well in this order - (a) the rollno for which data set needs to be generated, (b) the dimension of the data plot to be generated, and (c) the name and/or the path under which the plot is to be saved. For example,

sh elbow_plot2.sh AIB232065 4 /home/baadalvm/q3_4_2023AIB2065.png

'elbow_plot2.sh' uses 'generateDataset_d_dim_hpc_compiled', the precompiled file to generate data set given a roll number and the dimesion. 'script.py' is used to make the elbow plot again and save it in the given folder with the given name.


iii. Team mates and contribution

Suditi Laddha       2023AIB2065     33%
Anant Kumar Sah     2023AIB2068     33%
Ajay Kumar Meena    2023AIB2083     33%