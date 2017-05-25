from provider import Provider
from sklearn import svm

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
	for d in provider.getTestData():
		input.append(d[0])
		target.append(d[1][0])
	results = regressor.predict(input)
	w = open('output3_2.txt', 'w')
	for i, line in enumerate(results):
		w.write('%f;%f\n' % (target[i]*mult, line*mult))
	w.close()