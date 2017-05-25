from provider import Provider
from feature_selection import Selector
import numpy as np
import neurolab as nl
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    provider = Selector()
    net = nl.net.newff(provider.getDataRanges(), [provider.getInputCount()*2, provider.getInputCount()*2, 1])
    # net.trainf = nl.train.train_gd
    input = []
    target = []
    mult = provider.provider.multiplier
    for d in provider.getLearnData():
        input.append(d[0])
        target.append([d[1][0]/mult])
    err = net.train(input, target, epochs=1000, show=10, goal=0.001)#, lr=0.000001)

    results = net.sim(input)
    w = open('output1.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i][0]*mult, line[0]*mult))
    w.close()

    input = []
    target = []
    for d in provider.getTestData():
        input.append(d[0])
        target.append([d[1][0]/mult])
    results = net.sim(input)
    w = open('output2.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i][0]*mult, line[0]*mult))
    w.close()

    net.save('network8.net')
    # 1 - bpnn, 1 layer
    # 2 - simple train, 1 layer hidden
    # 3 - simple train, 2x2 layer hidden
    # 4 - simple train, 3x2 layer hidden <-- looks best
    # 5 - simple train, 3x3
    # 6 - less inputs, 3x3
    # 7 - simple, 1, feature-selected
    # 7 - simple, 2x2, feature-selected