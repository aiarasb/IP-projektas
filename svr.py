from provider import Provider
from sklearn import svm
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    provider = Provider()
    input = []
    target = []
    for d in provider.getLearnData():
        input.append(d[0])
        target.append(d[1][0])

    regressor = svm.NuSVR()
    #svm.SVR()
    #svm.NuSVR() - best
    #svm.LinearSVR() - worst
    regressor.fit(input, target)
    
	#print(regressor.coef_)

    mult = provider.multiplier

    results = regressor.predict(input)
    print(results)
    w = open('output3_1.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()
    
    input = []
    target = []
    tr = []
    for d in provider.getTestData():
        input.append(d[0])
        target.append(d[1][0])
    results = regressor.predict(input)
    w = open('output3_2.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
        tr.append(target[i]*mult)
    w.close()
    w = open('svr_mse.txt', 'w')
    w.write('%f;\n' % sqrt(mean_squared_error(target, results)))
    w.close()
    plt.plot(target, 'b', results, 'r')
    plt.ylabel('Reikšmė')
    plt.xlabel('Masyvo elementas')
    plt.title('Support vector regression grafikas')
    plt.show()