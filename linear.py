from dimension_reduction import Reducer
from feature_selection import Selector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # reducer = Reducer()
    reducer = Selector()
    input = []
    target = []
    for d in reducer.getLearnData():
        input.append(d[0])
        target.append(d[1][0])

    regressor = LinearRegression()
    regressor.fit(input, target)
    print(regressor.coef_)

    mult = 1#reducer.provider.multiplier

    results = regressor.predict(input)
    print(results)
    w = open('output2_1_s.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()

    input = []
    target = []
    for d in reducer.getTestData():
        input.append(d[0])
        target.append(d[1][0])
    results = regressor.predict(input)
    w = open('output2_2_s.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()
    w = open('linear_mse.txt', 'w')
    w.write('%f;\n' % sqrt(mean_squared_error(target, results)))
    w.close()
    plt.plot(target, 'b', results, 'r')
    plt.ylabel('Reikšmė')
    plt.xlabel('Masyvo elementas')
    plt.title('Multiple linear regression grafikas')
    plt.show()