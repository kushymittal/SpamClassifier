class emailClass(object):

	def __init__(self,training_data):
		self.training_data = training_data
		self.m_likelihood = {}
		self.b_likelihood= {}

		def words():
			words = 0
			for doc in self.training_data:
				for key in doc.dictionary:
					words +=doc.dictionary[key]
			return words

		self.total_words = words()

if __name__ == '__main__':
	main()
		