from numpy import * 
from numberClass import *
from training import *
from testing import *

classes = (10)

def readTrainingLabels():
	# Open The File
	file = open("facedata/facedatatrainlabels.txt" , "r")

	# Count each face in training set
	list = [0,0]
	labels = []

	# Total num of faces
	counter = 0

	for line in file:
		list[int(line.strip())]+=1
		counter+=1
		labels.append(int(line.strip()))

	file.close()

	return [float(x)/counter for x in list],labels

def readTrainingImages():
	# Open the Training Images
	file = open("facedata/facedatatrain.txt", "r")

	# The List to store the images
	list = []

	curr_image = zeros((70,60))
	i = 0


	for line in file:
		# Remove the \n
		line = line.rstrip()

		j = 0

		for character in line:
			if line[j] != ' ':
				curr_image[(i)%70][j] = 1
			j+=1

		i+=1

		if (i%70) == 0:
			list.append(curr_image)
			curr_image = zeros((70,60))

	return list

def readTestingLabels():
	# Open The File
	file = open("facedata/facedatatestlabels.txt" , "r")

	# Count each face in training set
	list = [0,0]
	labels = []

	# Total num of digits
	counter = 0

	for line in file:
		list[int(line.strip())]+=1
		counter+=1
		labels.append(int(line.strip()))

	file.close()

	return list,labels

def readTestingImages():
	# Open the Training Images
	file = open("facedata/facedatatest.txt", "r")

	# The List to store the images
	list = []

	curr_image = zeros((70,60))
	i = 0


	for line in file:
		# Remove the \n
		line = line.rstrip()

		j = 0

		for character in line:
			if line[j] != ' ':
				curr_image[(i)%70][j] = 1
			j+=1

		i+=1

		if (i%70) == 0:
			list.append(curr_image)
			curr_image = zeros((70,60))

	return list


def main():
	#priorList = readTrainingLabels()
	imagesList = readTrainingImages()
	labelsList, labels = readTrainingLabels()

	classesList = []

	for x in xrange(0,2):
		classesList.append(numberClassFace(x))
		classesList[x].setPrior(labelsList[x])

	for x in xrange(0,len(imagesList)):
		classesList[labels[x]].addTrainingData(imagesList[x])



	for i in xrange(1,2):
		for x in xrange(0,2):
			classesList[x].empirical_likelihood = smoothed_likelihood_face(classesList[x].training_data,i)

		testingImagesList = readTestingImages()
		hypotheticalLabels = []
		confusionMatrix = zeros((2,3))

		for x in xrange(0, len(testingImagesList)):
			hypotheticalLabels.append(numClassifierFace(classesList,testingImagesList[x]))

		hypotheticalClasses = [0,0]
		for element in hypotheticalLabels:
			hypotheticalClasses[element]+=1

		#print hypotheticalClasses
		
		testClasses, testLabels = readTestingLabels()

		error = list(array(hypotheticalLabels) - array(testLabels))

		error_by_class = []
		for x in xrange(0,2):
			error_by_class.append(100 - abs(float(hypotheticalClasses[x]-testClasses[x])*100/testClasses[x]))

		# Find the confusion matrix
		for x in xrange(0,len(testLabels)):
			confusionMatrix[testLabels[x]][hypotheticalLabels[x]] += 1

		for x in xrange(0,2):
			for y in xrange(0,2):
				confusionMatrix[x][y] = confusionMatrix[x][y] * 100 / testClasses[x]

		error_value = float(count_nonzero(error))/10

		#for i in xrange(0,10):
		#	print "Digit Class: ", i
		#	print "Highest Posterior", classesList[i].highestPosterior
		#	print classesList[i].highPostImage
		#	print "Lowest Posterior", classesList[i].lowestPosterior
		#	print classesList[i].lowPostImage
		print "The error is ", error_value
		print "Success Rate: ", float(100-error_value), " for a value of k: ", i

		#print "Classification Rate: ", error_by_class

		oddsRatio(classesList[1], classesList[0])

		print confusionMatrix
		#print "This is the priors: ",labelsList, " for a smoothing of: ", i
		#print "This is the actual stats: ",testClasses, " for a smoothing of: ", i
		#print "This is the hypothetical stats: ",hypotheticalClasses, " for a smoothing of: ", i
		print "Success By Digit: ", error_by_class
		#print "hypotheticalLabels: ", hypotheticalLabels
		#print "Test Labels: ", testLabels 
		#print "This is the likelihood: ", classesList[x].empirical_likelihood


if __name__ == '__main__':
	main()