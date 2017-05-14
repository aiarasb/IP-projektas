import csv

class Provider:
    dataCols = ['RB050', 'AGE', 'RB090', 'PB190', 'PE010', 'PE030', 'PH030', 'PL031', 'PL190']
    resultCols = ['PY010G', 'PY050G']
    data = []
    rawData = []
    multiplier = 1

    def __init__(self):
        file = open('PGS 2015 asmenys.csv')
        reader = csv.reader(file, delimiter=',')
        self.header = reader.__next__()
        dataIndexMap = []
        resultIndexMap = []
        for col in self.dataCols:
            dataIndexMap.append(self.header.index(col))
        for col in self.resultCols:
            resultIndexMap.append(self.header.index(col))
        data = []
        for row in reader:
            self.rawData.append(row)
            rowData = [[], [0.0]]
            for index in dataIndexMap:
                d = 0.0
                if len(row[index]) > 0:
                    d = float(row[index])
                rowData[0].append(d)
            for index in resultIndexMap:
                if len(row[index])>0:
                    rowData[1][0] = rowData[1][0] + float(row[index])   
            data.append(rowData)
        data2 = []
        for d in data:
            if d[1][0] > 0:
                data2.append(d)
                if d[1][0] > self.multiplier:
                    self.multiplier = d[1][0]
        for d in data2:
            d[1][0] = d[1][0] / self.multiplier
            self.data.append(d)

    def getLearnData(self):
        return self.data[:int(len(self.data)/3)]

    def getValidationData(self):
        return self.data[int(len(self.data)/3):int((2*len(self.data))/3)]

    def getTestData(self):
        return self.data[int((2*len(self.data))/3):]

    def getInputCount(self):
        return len(self.dataCols)

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

if __name__ == '__main__':
    provider = Provider()
    print(len(provider.getLearnData()))
    print(len(provider.getValidationData()))
    print(len(provider.getTestData()))
    print(provider.getDataRanges())
    print(provider.multiplier)
    print(provider.data[0])