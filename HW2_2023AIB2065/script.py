import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

if len(sys.argv) != 4:
    print("Usage: python script.py input_file dimension plot_name.png")
    sys.exit(1)

input_file = sys.argv[1]
dimension = int(sys.argv[2])
plot_name = sys.argv[3] 

with open(input_file, "r") as file:
    lines = file.readlines()

data = []
for line in lines:
    values = [float(x) for x in line.strip().split()][:dimension] 
    data.append(values)

data = np.array(data)

wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 16), wcss)
# plt.plot(4,wcss[-1], 's')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.savefig(plot_name)  
