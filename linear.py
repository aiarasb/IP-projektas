from dimension_reduction import Reducer
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    reducer = Reducer()
    input = []
    target = []
    for d in reducer.getLearnData():
        input.append(d[0])
        target.append(d[1][0])

    regressor = LinearRegression()
    regressor.fit(input, target)
    print(regressor.coef_)

    mult = reducer.provider.multiplier

    results = regressor.predict(input)
    print(results)
    w = open('output2_1_r.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()

    input = []
    target = []
    for d in reducer.getTestData():
        input.append(d[0])
        target.append(d[1][0])
    results = regressor.predict(input)
    w = open('output2_2_r.txt', 'w')
    for i, line in enumerate(results):
        w.write('%f;%f\n' % (target[i]*mult, line*mult))
    w.close()