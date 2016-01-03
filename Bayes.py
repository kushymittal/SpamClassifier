from emailClass import *
from numpy import *

mk = 1
bk = 1

def multinomial(classific):
	# number of times word appears in document/ documents in class

	for doc in classific.training_data:

		# for each word in the doc
		for key,value in doc.dictionary.iteritems():
			
			if(key not in classific.m_likelihood):
				counter = 0

				# for each doc in the training data
				for doc in classific.training_data:
					if(key in doc.dictionary):
						counter +=doc.dictionary[key]

				classific.m_likelihood[key] = float(counter + mk)/(2*mk+classific.total_words)

def bernouilli(classific):
	# number of documents word appears/ documents in class
	# for every doc in the training data
	for doc in classific.training_data:

		# for each word in the doc
		for key,value in doc.dictionary.iteritems():
			
			if(key not in classific.b_likelihood):
				counter = 0

				# for each doc in the training data
				for doc in classific.training_data:
					if(key in doc.dictionary):
						counter +=1

				classific.b_likelihood[key] = float(counter+bk)/(2*bk +len(classific.training_data))

def multinomial_map_estimate(testdoc, emailclass, otherclass):
	map_estimate = 0.0
	for key in testdoc.dictionary:
		if(key in emailclass.m_likelihood):
			map_estimate = float(map_estimate) + math.log(emailclass.m_likelihood[key])
		elif(key in otherclass.m_likelihood):
			guess = float(mk)/emailclass.total_words
			map_estimate += math.log(guess)

	return map_estimate


def multinomial_classifier(testdoc, spamclass, regclass):

	list = []
	map_spam = multinomial_map_estimate(testdoc, spamclass, regclass)
	map_reg = multinomial_map_estimate(testdoc, regclass, spamclass)

	
	list.append(map_reg)
	list.append(map_spam)

	testdoc.set_label(list.index(max(list)))

	return max(map_reg, map_spam), list.index(max(list))

def bernouilli_map_estimate(testdoc, emailclass, otherclass):
	map_estimate = 0.0
	for key in testdoc.dictionary:
		if(key in emailclass.b_likelihood):
			map_estimate += float(math.log(emailclass.b_likelihood[key]))
		elif(key in otherclass.b_likelihood):
			guess = float(bk)/emailclass.total_words
			map_estimate += math.log(guess)

	return map_estimate


def bernouilli_classifier(testdoc, spamclass, regclass):

	list = []
	map_spam = bernouilli_map_estimate(testdoc, spamclass, regclass)
	map_reg = bernouilli_map_estimate(testdoc, regclass, spamclass)

	list.append(map_reg)
	list.append(map_spam)

	testdoc.set_label(list.index(max(list)))

	return max(list), list.index(max(list))
	

if __name__ == '__main__':
	main()