from provider import Provider
import numpy as np
import neurolab as nl

if __name__ == '__main__':
    provider = Provider()
    net = nl.net.newff(provider.getDataRanges(), [provider.getInputCount(), 1])
    # net.trainf = nl.train.train_gd
    input = []
    target = []
    for d in provider.getLearnData():
        input.append(d[0])
        target.append(d[1])
    err = net.train(input, target, epochs=5000, show=10, goal=0.001)#, lr=0.000001)
    
    mult = provider.multiplier

    results = net.sim(input)
    w = open('output1.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i][0]*mult, line[0]*mult))
    w.close()

    input = []
    target = []
    for d in provider.getTestData():
        input.append(d[0])
        target.append(d[1])
    results = net.sim(input)
    w = open('output2.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i][0]*mult, line[0]*mult))
    w.close()

    net.save('network2.net')