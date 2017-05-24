from feature_selection import Selector
from sklearn.linear_model import LinearRegression

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

class Linear:
    def __init__(self, provider):
        self.provider = provider
        input = []
        target = []
        for d in self.provider.getLearnData():
            input.append(d[0])
            target.append(d[1][0])

        self.regressor = LinearRegression()
        self.regressor.fit(input, target)

    def test(self, test_no):
        input = []
        target = []
        if (test_no == 2):
            data = self.provider.getValidationData()
        else:
            data = self.provider.getTestData()
        
        for d in data:
            input.append(d[0])
            target.append(d[1][0])
        res = self.regressor.predict(input)
        results = []
        for i, line in enumerate(res):
            results.append([target[i], line])
        return results