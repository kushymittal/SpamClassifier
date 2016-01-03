class Document(object):

	def __init__(self, dictvalue, labelvalue = 2):
		self.label = labelvalue
		self.dictionary = dictvalue

	def set_label(self, labelvalue):
		self.label = labelvalue

if __name__ == '__main__':
	main()
		