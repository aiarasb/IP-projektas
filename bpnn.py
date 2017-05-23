from provider import Provider
import numpy as np
import neurolab as nl

if __name__ == '__main__':
    provider = Provider()
    net = nl.net.newff(provider.getDataRanges(), [provider.getInputCount()*3, provider.getInputCount()*3, provider.getInputCount()*3, 1])
    # net.trainf = nl.train.train_gd
    input = []
    target = []
    for d in provider.getLearnData():
        input.append(d[0])
        target.append(d[1])
    err = net.train(input, target, epochs=1000, show=10, goal=0.001)#, lr=0.000001)
    
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

    net.save('network6.net')
    # 1 - bpnn, 1 layer
    # 2 - simple train, 1 layer hidden
    # 3 - simple train, 2x2 layer hidden
    # 4 - simple train, 3x2 layer hidden <-- looks best
    # 5 - simple train, 3x3
    # 6 - less inputs, 3x3