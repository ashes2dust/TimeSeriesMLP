import csv

import matplotlib.pyplot as plt

path = "../../../data/relu/part-00000-3ac60f01-fc3b-46c5-bf5e-81f64645e3a3-c000.csv"

csv_fp = csv.reader(open(path))

Y1 = []
Y2 = []

i = 0
for line in csv_fp:
    i += 1
    if i == 1:
        continue
    Y1.append(float(line[0]))
    Y2.append(float(line[1]))

# m = max(max(Y1), max(Y2))
#
# for i in range(0, len(Y1)):
#     Y1[i] /= m
#     Y2[i] /= m

X = range(0, len(Y1))

plt.title('relu')

plt.plot(X, Y1, label='true', lw=1)

plt.plot(X, Y2, label='prediction', lw=1)

plt.legend()

plt.savefig('../../../img/relu_scaled.png')

plt.show()
