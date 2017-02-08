#!/usr/bin/py

import numpy as np
from scipy import stats

n = int(raw_input().strip())
line = [int(i.strip()) for i in raw_input().strip().split(" ")]
mean = np.average(line)
print mean
median = np.median(line)
print median
line.sort()
count = 0
max_count = 0
begin_num = line[0]
mode_list = []

for i in range(len(line)):
    current_num = line[i]
    if begin_num == current_num:
        count = count + 1
    else:
        if max_count < count:
            max_count = count
            mode = begin_num
        begin_num = current_num
        count = 1

print mode
print np.std(line)
#print stats.norm.interval(0.05, loc=mean, scale=np.std(line))
conf = stats.t.interval(0.95, len(line)-1, loc=np.mean(line), scale=stats.sem(line))
low = mean - 1.96*(np.std(line)/np.sqrt(n))
high = mean + 1.96*(np.std(line)/np.sqrt(n))
print low, high