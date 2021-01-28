import numpy as np

class Dataset(object):

    def __init__(self, X=None, y=None):
        #if X is None: X = []
        #if y is None: y = []
        #assert len(X) == len(y), "shapes are not equal"
        self.data = list(zip(X, y))

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
        x,y = self.data[self.last_entry_id]
        x = np.expand_dims(np.array(x), axis=0)
        return x, np.array(y)

    def format_sklearn(self):
        X, y = zip(*self.get_labeled_entries())
        #return np.array(X), np.array(y)
        return X, np.array(y)

    def get_entries(self):
        return self.data

    def get_labeled_entries(self):
        return list(filter(lambda entry: entry[1] is not None, self.data))

    def get_unlabeled_entries(self):
        return [
            (idx, entry[0]) for idx, entry in enumerate(self.data)
            if entry[1] is None
        ]






