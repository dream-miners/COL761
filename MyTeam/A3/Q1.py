import random
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
import time
import numpy as np
from statistics import mean
from scipy.spatial.distance import cdist
import sys

start = time.time()
# Defining the ranges of the values used by us
total_points = 10**6  # Adjust this number based on your time constraints
dim_range = [1, 2, 4, 8, 16, 32, 64]
batch = 100

# Generating the random points. Indexed by math.log2(dim)
random_all = {int(math.log2(dim)): [round(random.random(), 5) for _ in range(dim)] for dim in dim_range}

# Generating the values of all the distances for all the dimensions, after taking the samples
d1_list = []
d2_list = []
dinfi_list = []

for dim in dim_range:
    print(int(math.log2(dim)))
    list_all = random_all[int(math.log2(dim))]

    d1_list_individual = []
    d2_list_individual = []
    dinfi_list_individual = []

    for i in range(batch):
        point_1 = np.round(np.random.randn(total_points, dim), decimals=5)  # Generate random points around the original point

        # Check if the generated points are exactly the same as the reference point
        if np.array_equal(point_1, list_all):
            continue

        d1 = cdist(point_1, [list_all], metric='cityblock')  # L1 norm
        # d1 = np.sum(d1_l1)

        d2 = cdist(point_1, [list_all], metric='euclidean')  # L2 norm
        # d2 = np.sum(d2_l2)

        dinfi = cdist(point_1, [list_all], metric='chebyshev')  # Linf norm
        # dinfi = np.sum(dinfi_linf)

        min_d1 = np.min(d1)
        max_d1 = np.max(d1)

        min_d2 = np.min(d2)
        max_d2 = np.max(d2)

        min_dinfi = np.min(dinfi)
        max_dinfi = np.max(dinfi)

        
        if min_d1 == 0 or min_d2 == 0 or min_dinfi == 0:
            continue



        d1_list_individual.append(max_d1 / min_d1)
        d2_list_individual.append(max_d2 / min_d2)
        dinfi_list_individual.append(max_dinfi / min_dinfi)

    # Getting the minimum and maximum values in these lists
    d1_list.append(mean(d1_list_individual)/8)
    d2_list.append(mean(d2_list_individual)/8)
    dinfi_list.append(mean(dinfi_list_individual)/8)

# Making the plots and showing/storing them

plt.plot(dim_range, d1_list, 'r', dim_range, d2_list, 'b', dim_range, dinfi_list, 'g')
plt.xlabel('Dimensions of the points')
plt.ylabel('Average ratio of max to min distance')
plt.legend(['L1 dist', 'L2 dist', 'Linf dist'])


plt.title('Plot of average ratio v/s dimensions (Normal y-scale)')
plt.savefig('q1_plot.png')

plt.show()

end = time.time()
print("Running Time: ", (end - start)/60)