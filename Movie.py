from numpy import * 
from numberClass import *
from training import *
from testing import *
from Document import *
from emailClass import *
from Bayes import *
import operator

def readTrainingReviews():

	#Open the training emails
	file = open("sentiment/rt-train.txt", "r")

	emails = []
	

	for line in file:

		linelist = line.split()
		dictemails = {}


		for x in xrange(1,len(linelist)):
			a,b = linelist[x].split(":")
			dictemails[a] = int(b) 

		emails.append(Document(dictemails,labelvalue = int(linelist[0])))

	file.close()

	return emails

def readTestingReviews():

	#Open the training emails
	file = open("sentiment/rt-test.txt", "r")

	actuallabels = []
	testingemails= []
	

	for line in file:

		linelist = line.split()
		dictemails = {}

		for x in xrange(1,len(linelist)):
			a,b = linelist[x].split(":")
			dictemails[a] = int(b) 

		actuallabels.append(int(linelist[0]))
		testingemails.append(Document(dictemails))

	file.close()

	return actuallabels,testingemails

def main():
	training_emails = readTrainingReviews()
	neg_emails = []
	pos_emails = []

	for email in training_emails:
		if(email.label == 1):
			pos_emails.append(email)
		else:
			neg_emails.append(email)

	emails_classes = []
	emails_classes.append(emailClass(neg_emails))
	emails_classes.append(emailClass(pos_emails))
	
	multinomial(emails_classes[0])
	multinomial(emails_classes[1])
	bernouilli(emails_classes[0])
	bernouilli(emails_classes[1])

	file = open("multinomial_negative.txt", "w")
	print>>file, emails_classes[0].m_likelihood
	file.close()

	file = open("multinomial_positive.txt", "w")
	print>>file, emails_classes[1].m_likelihood
	file.close()

	file = open("bernouilli_negative.txt", "w")
	print>>file, emails_classes[0].b_likelihood
	file.close()

	file = open("bernouilli_positive.txt", "w")
	print>>file, emails_classes[1].b_likelihood
	file.close()

	actual_labels, testing_emails = readTestingReviews()
	confusionMatrixM = zeros((2,2))
	confusionMatrixB = zeros((2,2))

	hypothetical_m_classifier = []
	hypothetical_b_classifier = []
	map_m_estimate = []
	map_b_estimate = []

	for x in xrange(0,len(testing_emails)):
		map_estimate, label = multinomial_classifier(testing_emails[x], emails_classes[0], emails_classes[1])
		map_estimate2, label2 = bernouilli_classifier(testing_emails[x], emails_classes[0], emails_classes[1])
		hypothetical_m_classifier.append(label)
		hypothetical_b_classifier.append(label2)
		map_m_estimate.append(map_estimate)
		map_b_estimate.append(map_estimate2)

	count_m_list = []
	count_m_0 = len(hypothetical_m_classifier) - count_nonzero(hypothetical_m_classifier)
	count_m_1 = count_nonzero(hypothetical_m_classifier)
	count_m_list.append(count_m_0)
	count_m_list.append(count_m_1)

	count_b_list = []
	count_b_0 = len(hypothetical_b_classifier) - count_nonzero(hypothetical_b_classifier)
	count_b_1 = count_nonzero(hypothetical_b_classifier)
	count_b_list.append(count_b_0)
	count_b_list.append(count_b_1)

	temp_actual = actual_labels[:]
	for x in xrange(0,len(actual_labels)):
		if(actual_labels[x] == 1):
			temp_actual[x] = 0
		elif(actual_labels[x] == -1):
			temp_actual[x] = 1

	for x in xrange(0,len(temp_actual)):
		confusionMatrixM[temp_actual[x]][hypothetical_m_classifier[x]] += 1
		confusionMatrixB[temp_actual[x]][hypothetical_b_classifier[x]] += 1

	for x in xrange(0,2):
		for y in xrange(0,2):
			confusionMatrixM[x][y] = float(confusionMatrixM[x][y] * 100) / count_m_list[x]
			print count_m_list[x]
			confusionMatrixB[x][y] = float(confusionMatrixB[x][y] * 100) / count_b_list[x]

	print confusionMatrixM
	print confusionMatrixB

	for x in xrange(0,len(hypothetical_m_classifier)):
		if(hypothetical_m_classifier[x] == 1):
			hypothetical_m_classifier[x] = -1
		elif(hypothetical_m_classifier[x] == 0):
			hypothetical_m_classifier[x] = 1

		if(hypothetical_b_classifier[x] == 1):
			hypothetical_b_classifier[x] = -1
		elif(hypothetical_b_classifier[x] == 0):
			hypothetical_b_classifier[x] = 1

	error = list(array(actual_labels) - array(hypothetical_m_classifier))
	error_b = list(array(actual_labels) - array(hypothetical_b_classifier))

	error_value = float(count_nonzero(error))/len(testing_emails)
	error_b_value = float(count_nonzero(error_b))/len(testing_emails)

	sorted_m_positive = sorted(emails_classes[0].m_likelihood.items(), key=operator.itemgetter(1), reverse = True)
	sorted_m_negative = sorted(emails_classes[1].m_likelihood.items(), key=operator.itemgetter(1), reverse = True)
	sorted_b_positive = sorted(emails_classes[0].b_likelihood.items(), key=operator.itemgetter(1), reverse = True)
	sorted_b_negative = sorted(emails_classes[1].b_likelihood.items(), key=operator.itemgetter(1), reverse = True)

	for x in xrange(0,20):
		print "Word: ", sorted_m_negative[x][0], " Value: ", sorted_m_negative[x][1]

	print "Booyah"
	for x in xrange(0,20):
		print "Word: ", sorted_b_positive[x][0], " Value: ", sorted_b_positive[x][1]
	print "Huzzah!"
	for x in xrange(0,20):
		print "Word: ", sorted_b_negative[x][0], " Value: ", sorted_b_negative[x][1]
	

	print "The multinomial error is ", error_value
	
	print "The bernouilli error is ", error_b_value
	
	file = open("map_m_estimate.txt", "w")
	print>>file, map_m_estimate
	file.close()

	file = open("map_b_estimate.txt", "w")
	print>>file, map_b_estimate
	file.close()

	file = open("hypothetical_m_classifier.txt", "w")
	print>>file, hypothetical_m_classifier
	file.close()

	file = open("hypothetical_b_classifier.txt", "w")
	print>>file, hypothetical_b_classifier
	file.close()

	file = open("actual_labels.txt", "w")
	print>>file, actual_labels
	file.close()

if __name__ == '__main__':
	main()