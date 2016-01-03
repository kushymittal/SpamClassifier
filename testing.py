from numpy import *
from numberClass import *
import matplotlib.pyplot as plt

"""
This function computes the mapEstimate (posterior)
for a testimage and a potential class
testData = the test image to be classified
possibleNumber = numberClass object to which
		   this testimage might belong
"""
def mapEstimate(possibleNumber, testData):

	prior = possibleNumber.prior

	computedLikelihoods = possibleNumber.empirical_likelihood

	total_likelihood = 0

	for x in xrange(0,28):
		for y in xrange(0,28):
			if testData[x][y] == 0:
				total_likelihood += math.log(1 - computedLikelihoods[x][y])
			else:
				total_likelihood += math.log(computedLikelihoods[x][y])

	return math.log(prior) + total_likelihood

"""
This function assignes a testimage to the class
with the highest posterior
classList = an array of numberClass objects
testData = the input test image
"""
def numClassifier(classList, testData):
	list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	for x in xrange(0,10):
		list[x] = mapEstimate(classList[x], testData)

	assignedClass = list.index(max(list))

	return assignedClass

"""
This function computes the mapEstimate (posterior)
for a testimage and a potential class
testData = the test image to be classified
possibleNumber = numberClass object to which
		   this testimage might belong
"""
def mapEstimateFace(possibleNumber, testData):

	prior = possibleNumber.prior

	computedLikelihoods = possibleNumber.empirical_likelihood

	total_likelihood = 0

	for x in xrange(0,70):
		for y in xrange(0,60):
			if testData[x][y] == 0:
				total_likelihood += math.log(1 - computedLikelihoods[x][y])
			else:
				total_likelihood += math.log(computedLikelihoods[x][y])

	return math.log(prior) + total_likelihood

"""
This function assignes a testimage to the class
with the highest posterior
classList = an array of numberClass objects
testData = the input test image
"""
def numClassifierFace(classList, testData):
	list = [0, 0]

	for x in xrange(0,2):
		list[x] = mapEstimateFace(classList[x], testData)

	assignedClass = list.index(max(list))

	return assignedClass

"""
This function print the heat map associated
with an imaage based on it's likelihoods 
"""
def oddsRatio(class1, class2):
	#temp = class1.empirical_likelihood/class2.empirical_likelihood
	temp = zeros((28,28))
	for x in xrange(0,28):
		for y in xrange(0,28):
			temp[x][y] = math.log(class1.empirical_likelihood[x][y]) - math.log(class2.empirical_likelihood[x][y])


	img1 = plt.imshow(temp)
	plt.set_cmap('RdYlBu')
	plt.colorbar()
	plt.savefig('oddsRatio.png')

	img2 = plt.imshow(class1.empirical_likelihood)
	plt.savefig('empirical_likelihood_1.png')

	img3 = plt.imshow(class2.empirical_likelihood)
	plt.savefig('empirical_likelihood_2.png')


	




