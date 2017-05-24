from provider import Provider
from feature_selection import Selector
from dimension_reduction import Reducer
from sklearn import tree

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
	for d in provider.getTestData():
		input.append(d[0])
		target.append(d[1][0])
	results = regressor.predict(input)
	w = open('output4_2_sr.txt', 'w')
	for i, line in enumerate(results):
		w.write('%f;%f\n' % (target[i]*mult, line*mult))
	w.close()