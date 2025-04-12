###############################################################################
# Script for generating results from paper
###############################################################################

import subprocess
import matplotlib.pyplot as plt
import os, sys

sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
results = {i : [] for i in sizes}

for i in sizes:
    print("Running with step size %d" % i) 
    c = subprocess.run([os.path.join(os.getcwd(), sys.argv[1]), "%d" % i,
                        "1000000000"], stdout = subprocess.PIPE)
    print(c.stdout.decode('ascii'))
    results[i].append(float(c.stdout.decode('ascii').split()[-1]))
    
    
plt.plot(sizes, list(results.values()))    
#plt.legend()
plt.title("Update Every K-th Int")
plt.xlabel("K")
plt.ylabel("Time(ms)")
plt.xscale('log', basex=2)
plt.xticks(sizes, sizes)
plt.show()
