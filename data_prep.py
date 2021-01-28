import numpy as np
import pandas as pd
from feature_mod import *

FILE = 'data/movie_reviews.csv'

MAX_LEN = 400 #only for GPT2 and BERT models
F_TYPES = ['use', 'gpt2', 'bert']

if __name__ == '__main__':

	for f in F_TYPES:
		df = pd.read_csv(FILE)

		if f == 'w2v':
			features = W2VFeaturizer().get_vectors(df['documents'].tolist())
			df['document_features'] = features

		elif f == 'use':
			tfhub = TFHubExtract()
			df['document_features'] = tfhub.predict(df['documents'].tolist()).tolist()

		else:
			featurizer = Featurizer(f)
			df['documents'] = df['documents'].apply(lambda x: ' '.join(x.split(' ')[:MAX_LEN]))
			features = []
			for x in df['documents'].tolist():
				try:
					features.append(featurizer.get_feature(x))
				except:
					print("Didn't work")
					features.append(np.nan)

			df['document_features'] = features
			df.dropna(inplace=True)	

		df.to_pickle(FILE.split('.')[0] + '_{}.pkl'.format(f))





