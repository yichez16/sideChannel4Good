
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import *
from pylab import *
data = 'CUPTI_counter.csv'
df = pd.read_csv(data,
                  names=["0", "1" , "2", "3", "Time", "5"])
df['Time'] = (df['Time'] -  df['Time'][0])/1000000
df = df.iloc[3:]

counter0 = df['0']
time = df['Time']
plt.plot(time, counter0)


plt.title('Footprint of Modified MatMul with 10 Hz Impulses', fontweight="bold", y =1.05)
plt.xlabel('Time (s)')
# plt.legend(title='Instructions per Impulse', bbox_to_anchor=(1, 1))

plt.ylabel('active_warps',  fontweight="bold")

plt.grid(True, linestyle='--')

plt.savefig('active_warps.jpg', dpi=200,  bbox_inches='tight')

plt.show()
