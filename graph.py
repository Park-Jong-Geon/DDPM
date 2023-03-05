import os
import matplotlib.pyplot as plt

# text_name = 'train_log.txt'
text_name = 'test.txt'

f = open(text_name)
lines = f.readlines()

loss = []
for line in lines:
    if line == '\n':
        continue
    strings = line.split()
    #print(f"strings : {strings}")
    
    if strings[0] == 'Loss':
        loss.append(float(strings[-1]))

#print(loss)

partition = text_name.split('.')

plt.plot(loss)
plt.axis((0, 1000, 0.025, 0.03))
plt.savefig(f"{partition[0]}_loss_curve.png")