from transformers import GPT2Model, GPT2Tokenizer, BertTokenizer, BertModel, pipeline
import numpy as np
from gensim.models import KeyedVectors
import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

class TFHubExtract(object):

	def __init__(self, path="https://tfhub.dev/google/universal-sentence-encoder/4"):
		import tensorflow as tf
		import tensorflow_hub as hub

		# Create graph and finalize (finalizing optional but recommended).
		g = tf.Graph()
		with g.as_default():
			# We will be feeding 1D tensors of text into the graph.
			self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
			embed = hub.Module(path)
			self.embedded_text = embed(self.text_input)
			init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
		g.finalize()
		self.session = tf.Session(graph=g)	
		self.session.run(init_op)

	def predict(self, x):
		return self.session.run(self.embedded_text, feed_dict={self.text_input: x})

class Featurizer(object):

	def __init__(self, model_name):
		self.model_name = model_name

		if self.model_name == 'gpt2':
			self.model = GPT2Model.from_pretrained('gpt2-medium', return_dict=True)
			self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
			self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]'})

		elif self.model_name == 'bert':
			self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]'})

		self.nlp_features = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer)

	def get_feature(self, doc):
		return np.mean(np.squeeze(np.array(self.nlp_features(doc))), axis=0).tolist()

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = dim

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class W2VFeaturizer(object):

	def __init__(self):
		self.STOP_WORDS = list(ENGLISH_STOP_WORDS)
		word2vec_model = "GoogleNews-vectors-negative300.bin"
		self.W2V_MODEL = KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
		self.embedding = MeanEmbeddingVectorizer(self.W2V_MODEL , 300)

	def text_prep(self, docs):
		def check_word(word):
			if (word not in self.STOP_WORDS and len(word)>1 and re.match(r"^[a-z]+$", word)!=None):
				if word in self.W2V_MODEL:
					return True
			return False  
		req_text = []
		for t in docs:
			req_text.append([word for word in t.split() if check_word(word)])
		return req_text

	def clean_str(self, s):
		"""Clean sentence"""
		s = re.sub(r"[^A-Za-z]", " ", s)
		return s.strip().lower()

	def get_vectors(self, docs):
		docs = [self.clean_str(x) for x in docs]
		docs = self.text_prep(docs)
		return self.embedding.transform(docs).tolist()


