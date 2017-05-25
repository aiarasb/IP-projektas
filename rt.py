from provider import Provider
from feature_selection import Selector
from dimension_reduction import Reducer
from sklearn import tree
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    provider = Selector()
    input = []
    target = []
    for d in provider.getLearnData():
        input.append(d[0])
        target.append(d[1][0])
    regressor = tree.DecisionTreeRegressor()
    regressor.fit(input,target)

    mult = 1#provider.multiplier
	
    results = regressor.predict(input)
    print(results)
    w = open('output4_1_sr.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()

    input = []
    target = []
    etrue = []
    epredict = []
    for d in provider.getTestData():
        input.append(d[0])
        target.append(d[1][0])
    results = regressor.predict(input)
    w = open('output4_2_sr.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()
    w = open('rt_mse.txt', 'w')
    w.write('%f;\n' % sqrt(mean_squared_error(target, results)))
    w.close()
    plt.plot(target, 'b', results, 'r')
    plt.ylabel('Reikšmė')
    plt.xlabel('Masyvo elementas')
    plt.title('Regression tree grafikas')
    plt.show()