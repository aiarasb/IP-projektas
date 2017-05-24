from provider import Provider
import numpy as np
from sklearn.decomposition import PCA

class Reducer:
    provider = Provider()
    data = []

    def __init__(self):
        data = self.provider.data
        inputData = []
        targetData = []
        for d in data:
            inputData.append(d[0])
            targetData.append(d[1])
        X = np.array(inputData)
        pca = PCA(n_components=10)
        XX = pca.fit_transform(X)
        for i, t in enumerate(targetData):
            self.data.append([XX[i], t])

    def getLearnData(self):
        return self.data[:int(len(self.data)/3)]

    def getValidationData(self):
        return self.data[int(len(self.data)/3):int((2*len(self.data))/3)]

    def getTestData(self):
        return self.data[int((2*len(self.data))/3):]