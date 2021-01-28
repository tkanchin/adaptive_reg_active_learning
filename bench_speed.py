import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timeit
from sklearn.linear_model import Ridge

from efficient_learning.model import *

FILE = 'data/movie_reviews_use.pkl'

### load data
def load_data():
	df = pd.read_pickle(FILE)
	
	X = np.array(df["document_features"].tolist())
	Y = np.array(df["labels"].tolist())
	
	X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=64)	
	return X_tr, X_te, Y_tr, Y_te

if __name__ == '__main__':

	X_tr, X_te, Y_tr, Y_te = load_data()

	## sklearn Ridge
	start_time = timeit.default_timer()
	clf = Ridge(alpha=0.001)
	for i in range(1, X_tr.shape[0]):
		clf.fit(X_tr, Y_tr)
	print("Sklearn Ridge AUC : {}".format(roc_auc_score(Y_te, clf.predict(X_te))))
	print("Total time: {} s".format(timeit.default_timer() - start_time))
	print("#" * 10)

	## LS-SVM
	start_time = timeit.default_timer()
	clf = LSSVM(lambda_p=0.001)
	for i in range(1, X_tr.shape[0]):
		clf.fit(X_tr[:i], Y_tr[:i])
	clf.fit(X_tr, Y_tr)
	print("Full LS-SVM : {}".format(roc_auc_score(Y_te, clf.predict(X_te))))
	print("Total time: {} s".format(timeit.default_timer() - start_time))
	print("#" * 10)	

	## Incremental LS-SVM SMW
	start_time = timeit.default_timer()
	clf = LSSVM(lambda_p=0.001)
	clf.fit(X_tr[:2], Y_tr[:2])
	for i in range(2, X_tr.shape[0]):
		clf.fit_update(np.expand_dims(X_tr[i], axis=0), Y_tr[i])
	print("Incremental LS-SVM SMW : {}".format(roc_auc_score(Y_te, clf.predict(X_te))))
	print("Total time: {} s".format(timeit.default_timer() - start_time))
	print("#" * 10)	

	## Incremental LS-SVM Poly
	start_time = timeit.default_timer()
	clf = LSSVMPOLY(lambda_p=0.001)
	clf.fit(X_tr[:2], Y_tr[:2])
	for i in range(2, X_tr.shape[0]):
		clf.fit_update(np.expand_dims(X_tr[i], axis=0), Y_tr[i])
	print("Incremental LS-SVM Poly : {}".format(roc_auc_score(Y_te, clf.predict(X_te))))
	print("Total time: {} s".format(timeit.default_timer() - start_time))
	print("#" * 10)

	## Incremental LS-SVM SVD
	start_time = timeit.default_timer()
	clf = LSSVMSVD(lambda_p=0.001)
	clf.fit(X_tr[:2], Y_tr[:2])
	for i in range(2, X_tr.shape[0]):
		clf.fit_update(np.expand_dims(X_tr[i], axis=0), Y_tr[i])
	print("Incremental LS-SVM SVD : {}".format(roc_auc_score(Y_te, clf.predict(X_te))))
	print("Total time: {} s".format(timeit.default_timer() - start_time))
	print("#" * 10)





