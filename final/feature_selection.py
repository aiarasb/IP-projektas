import numpy
from provider import Provider
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

class Selector:
    data = []
    provider = Provider()
    outputs = 20

    def __init__(self):
        inputData = []
        targetData = []
        for d in self.provider.data:
            inputData.append(d[0])
            targetData.append(d[1][0])
        # feature extraction
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(targetData)
        test = SelectKBest(score_func=chi2, k=self.outputs)
        fit = test.fit(inputData, encoded)
        # summarize scores
        fields = zip(fit.scores_, self.provider.dataCols)
        self.sortedFields = sorted(fields, key=lambda x: x[0], reverse=True)[0:self.outputs]
        # print(sortedFields[0:self.outputs])
        features = fit.transform(inputData)
        for i, t in enumerate(targetData):
            self.data.append([features[i], [t]])
        self.multiplier = self.provider.multiplier

    def getLearnData(self):
        return self.data[:int(len(self.data)/3)]

    def getValidationData(self):
        return self.data[int(len(self.data)/3):int((2*len(self.data))/3)]

    def getTestData(self):
        return self.data[int((2*len(self.data))/3):]

    def getDataRanges(self):
        ranges = []
        for cel in self.data[0][0]:
            ranges.append([cel, cel])
        for row in self.data:
            for i, cel in enumerate(row[0]):
                if ranges[i][0] > cel:
                    ranges[i][0] = cel
                if ranges[i][1] < cel:
                    ranges[i][1] = cel
        return ranges

    def getInputCount(self):
        return self.outputs