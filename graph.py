import os
import matplotlib.pyplot as plt

text_name = 'cifar10_debug.txt'

f = open(text_name)
lines = f.readlines()

loss = []
for line in lines:
    strings = line.split()
    #print(f"strings : {strings}")
    if strings[0] == 'Loss':
        loss.append(float(strings[-1]))

#print(loss)

partition = text_name.split('.')

plt.plot(loss)
plt.savefig(f"{partition[0]}_loss_curve.png")