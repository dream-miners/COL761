import matplotlib.pyplot as plt
def return_time_values(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    time_values = []

    for line in lines:
        if line.startswith("real"):

            parts = line.split()
            time = parts[1]

            if 'm' in time:

                minutes = float(time.split('m')[0])
                seconds = float(time.split('m')[1][:-1])

                time_in_seconds = (minutes * 60) + seconds

            time_values.append(time_in_seconds)
            
    file.close()
    
    return time_values

time_gaston = return_time_values('timeGaston.txt')
time_gSpan = return_time_values('timeGspan.txt')
time_FSG = return_time_values('timeFSG.txt')
support = [5, 10, 25, 50, 95]
plt.plot(support,time_gaston)
plt.plot(support,time_gSpan)
plt.plot(support,time_FSG)
plt.xlabel('Support(%)')
plt.ylabel('Time (seconds)')
plt.title('Execution Time')
plt.legend(['Gaston','gSpan','FSG'])
plt.savefig('time.png')
plt.show()