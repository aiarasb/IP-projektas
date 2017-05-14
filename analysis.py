from provider import Provider
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    provider = Provider()
    data = provider.rawData
    header = provider.header
    resultCols = provider.resultCols
    dataCols = ['RB050', 'AGE', 'RB090', 'PB190', 'PE010', 'PE030', 'PH030', 'PL031', 'PL190']
    datas = {}
    counts = []
    res = []

    for col in dataCols:
        datas[col] = []

    dataIndexMap = {}
    resultIndexMap = []
    for col in dataCols:
        dataIndexMap[col] = header.index(col)
    for col in resultCols:
        resultIndexMap.append(header.index(col))

    for line in data:
        if int(line[dataIndexMap['AGE']]) >= 18: #AGE
            for col in dataCols:
                val = -1.0
                if len(line[dataIndexMap[col]]) > 0:
                    val = float(line[dataIndexMap[col]])
                datas[col].append(val)
            ress = 0.0
            for index in resultIndexMap:
                if len(line[index])>0:
                    ress = ress + float(line[index])
            res.append(ress)

    for i, d in enumerate(dataCols):
        if d not in ['RB050', 'AGE', 'PE030', 'PL190']:
            vals = {}
            counts = {}
            for xi, x in enumerate(datas[d]):
                p = int(x)
                if p in vals:
                    vals[p] += res[xi]
                    counts[p] += 1.0
                else:
                    vals[p] = res[xi]
                    counts[p] = 1.0
            p = []
            q = []
            for k in vals.keys():
                p.append(k)
                q.append(vals[k]/counts[k])
            plt.figure(i)
            plt.plot(p, q, 'ro')
            plt.title(d)
            plt.show()
        else:
            plt.figure(i)
            plt.plot(datas[d], res, 'ro')
            plt.title(d)
            plt.show()
