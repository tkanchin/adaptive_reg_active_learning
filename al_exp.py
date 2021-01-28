from sklearn.model_selection import train_test_split
from json_tricks import dumps, load
import pandas as pd
import numpy as np
import os, math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def kernel_func(x, y=None):
	if y is None:
		y = x
	return rbf_kernel(x, y, gamma=1.0)

class UncertaintySampling:
	def __init__(self, dataset, qc='uncertain'):
		self.dataset = dataset
		self.qc = qc
		self.model = LogisticRegression(solver='lbfgs')
		self.train()

	def train(self):
		X, Y = self.dataset.format_sklearn()
		self.model.fit(X, Y)

	def make_query(self):
		self.train()
		unlabeled_entry_ids, X = zip(*self.dataset.get_unlabeled_entries())
		if self.qc == 'random':
			return np.random.choice(unlabeled_entry_ids, 1, replace=False)[0]
		elif self.qc == 'uncertain':
			ask_id = np.argmax(-np.max(self.model.predict_proba(np.array(X)), axis=1))	
		else:
			uncertain_scores = -np.max(self.model.predict_proba(np.array(X)), axis=1)
			distance_scores = kernel_func(np.array(X)).min(axis=1)
			similarity_scores = 1 / (1 + distance_scores)
			alpha = len(unlabeled_entry_ids)/len(X)
			scores = alpha * (1 - similarity_scores) + (1 - alpha) * uncertain_scores
			ask_id = np.argmax(scores)				
		return unlabeled_entry_ids[ask_id]

	def get_auc(self, tst_ds):
		X, Y = tst_ds.format_sklearn()
		pred = self.model.decision_function(X)
		return roc_auc_score(Y, pred)

class Dataset(object):

	def __init__(self, X, y):
		self.data = list(zip(X, y))
		self.last_entry_id = None

	def __len__(self):
		return len(self.data)

	def len_labeled(self):
		return len(self.get_labeled_entries())

	def len_unlabeled(self):
		return len(list(filter(lambda entry: entry[1] is None, self.data)))

	def get_num_of_labels(self): #distinct labels
		return len({entry[1] for entry in self.data if entry[1] is not None})

	def update(self, entry_id, new_label):
		self.last_entry_id = entry_id
		self.data[entry_id] = (self.data[entry_id][0], new_label)

	def get_last_entry(self):
		X, y = self.data[self.last_entry_id]
		X = np.expand_dims(np.array(X), axis=0)
		return Xd, np.array(y)

	def format_sklearn(self):
		return self.get_labeled_entries()

	def get_entries(self):
		return self.data

	def get_labeled_entries(self):
		X, Y =  zip(*list(filter(lambda entry: entry[1] is not None, self.data)))
		return np.array(X), np.array(Y)
		
	def get_unlabeled_entries(self):
		return [
			(idx, entry[0]) for idx, entry in enumerate(self.data)
			if entry[1] is None
			]

def load_data(filename, n_labeled):
	df = pd.read_pickle(filename)
	#COLS = ['document', 'features', 'Machine_labels', 'Manual_labels']
	#df = df[COLS]
	#df.columns = ['document', 'features', 'pseudo', 'labels']
	df.dropna(inplace=True)

	df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)
	
	X_tr = np.array(df_train['document_features'].tolist())
	Y_tr = np.array(df_train["labels"].tolist())	

	X_te = np.array(df_test['document_features'].tolist())
	Y_te = np.array(df_test["labels"].tolist())		
	
	X_ul, X_l, Y_ul, Y_l = train_test_split(X_tr, Y_tr, test_size=n_labeled)


	print(X_tr.shape)

	while len(np.unique(Y_l)) < 2:
		X_ul, X_l, Y_ul, Y_l = train_test_split(X_tr, Y_tr, test_size=n_labeled)

	trn_ds = Dataset(np.concatenate((X_ul, X_l), axis=0),
		np.concatenate([[None] * (len(Y_tr) - n_labeled), Y_l]))

	tst_ds = Dataset(X_te, Y_te)
	return trn_ds, tst_ds, Y_ul

def active_learning_simulate(filename, qc='uncertain', n_labeled=2, n_test_rounds=5):

	global_auc_tracker = []

	for _ in range(n_test_rounds):
		trn_ds, tst_ds, labeler = load_data(filename, n_labeled)

		auc_tracker = []

		qs = UncertaintySampling(dataset=trn_ds, qc=qc)

		auc_tracker.append(qs.get_auc(tst_ds))
		
		num_labels = 2
		
		while(trn_ds.len_unlabeled() > 5):
			ask_id = qs.make_query()
			trn_ds.update(ask_id, labeler[ask_id])
			auc = qs.get_auc(tst_ds)
			auc_tracker.append(auc)
		global_auc_tracker.append(auc_tracker)

	return np.mean(global_auc_tracker, axis=0).tolist()

def append_str(lis, string):
	return map(lambda x: '{}_{}.pkl'.format(x.split('.')[0], string), lis)	

if __name__ == '__main__':

	FILE = 'data/movie_reviews_use.pkl'

	results = {}
	sampling_types = ['uncertain', 'random']

	for sampling in sampling_types:
		results['{}'.format(sampling)] = active_learning_simulate(FILE, qc=sampling)

	with open('results.json', 'w') as f:
		f.write(dumps(results))
