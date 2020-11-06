from math import exp
from numpy.random import Generator, MT19937
from scipy.spatial import distance
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np
import numpy.linalg
import pandas as pd


def inv(mat):
    return numpy.linalg.pinv(mat)




class tp_miner_t:
    def __init__(self, X, cov_mat, dist_threshold):
        self._X = X
        self._cm = inv(cov_mat)
        self._dt = dist_threshold

        self._typical_objs = list()
        self._supported_objs = set()


    def fit(self):
        self.search_typical_objects()
        return self._typical_objs


    def search_typical_objects(self):
        """ Search for typical objects in training set until all objects become
            a typical object or a supported object.
        """
        while len(self._supported_objs) < len(self._X):
            i, supported_objects = self.get_most_typical()
            most_tipical_object = self._X[i]

            # add the objects supported by most typical objects to supported set
            self._typical_objs.append(np.array(most_tipical_object))
            self._supported_objs.update(supported_objects)


    def get_most_typical(self):
        """ Eval the typicality of all object and return the most typical. """
        typicality = list()

        for i, obj in enumerate(self._X):
            if i in self._supported_objs: # ignore already supported objects
                continue

            closer_objects = self.get_closer_objects(obj)
            object_typicality = np.sum([exp(-0.5*(d/self._dt)**2) for _, d
                                        in closer_objects])

            typicality.append((object_typicality, i, closer_objects))

        typicality.sort(key=lambda obj:obj[0], reverse=True)

        most_typical = typicality[0]
        index = most_typical[1]
        supported_objects = set([i for i, dist in most_typical[2]])
        return index, supported_objects


    def get_closer_objects(self, obj):
        """Returns a numpy array whose elements are a tuple of index to training
           set and the correspondent distance.
        """
        closer_objects = list()

        for i, similar_obj in enumerate(self._X):
            if i in self._supported_objs: # ignore supported objects.
                continue

            dist = distance.mahalanobis(obj, similar_obj, self._cm)

            if dist <= self._dt: # closer than threshold.
                closer_objects.append((i, dist))

        return closer_objects




class BTPM:
    """ Implementation of Bagging for TPMiner """

    def __init__(self, N, sample_prop, q_size = 0, seed=25111996):
        """
           Parameters:
                training_set: Full training set
                n: Quantity of classifiers
                sample_prop: Proportion of training set to sample
                q_size: How many previous classification to consider
                seed: Seed for reproducibility
        """
        self._n = N
        self._ratio = sample_prop
        self._q_size = q_size
        self._rd = Generator(MT19937(seed))

        self._past_votes = list()
        self._params = list()


    def fit(self, X, y=None):
        """ Train if not yet trained or if forced """
        self.train_inner_classifiers(X)


    def train_inner_classifiers(self, X):
        """ Train each inner classifier """
        for i in range(self._n):
            local_X = self.bootstrap(X) # sample from dataset
            cov_mat = self.covariance_matrix(local_X)
            avg_dist = self.average_distance(local_X, cov_mat)
            tp_miner = tp_miner_t(local_X, cov_mat, avg_dist)
            typical_objects = tp_miner.fit()

            self._params.append((typical_objects, cov_mat, avg_dist))


    def predict(self, objs):
        if objs.ndim > 1:
            n_examples = objs.shape[0]
            predicted = np.zeros(shape=n_examples)
            for i in range(n_examples):
                predicted[i] = self.predict_one(objs[i])

            return predicted
        else:
            return self.predict_one(obj)
        
    def predict_one(self, obj):
        votes = list()
        for typical_objects, cov_mat, avg_dist in self._params:
            distances = list()
            inverse_cov_mat = inv(cov_mat)

            for ref_obj in typical_objects:
                dist = distance.mahalanobis(obj, ref_obj, inverse_cov_mat)
                distances.append(dist)

            min_dist = np.min(distances)
            vote = exp(-0.5*(min_dist/avg_dist)**2)
            votes.append(vote)

        avg_votes = np.mean(votes)
        avg_past_votes = self.avg_past_votes()

        self.update_past_votes(avg_votes)

        if self._q_size == 0:
            return avg_votes
        else:
            return (avg_votes + avg_past_votes) / 2


    def avg_past_votes(self):
        if self._q_size == 0 or len(self._past_votes) == 0: return 0
        return np.mean(self._past_votes)


    def update_past_votes(self, new_vote):
        self._past_votes.append(new_vote)
        self._past_votes = self._past_votes[:self._q_size]


    def bootstrap(self, X):
        n_examples = X.shape[0]
        n_samples = int(n_examples * self._ratio)
        samples = self._rd.choice(n_examples, size=n_samples, replace=True)
        return X[samples]


    def covariance_matrix(self, X):
        # observations are in rows and variables in columns (different from
        # default parameters).
        return np.cov(X.T)


    def average_distance(self, training_set, cov_mat):
        i_cov = inv(cov_mat)

        n = len(training_set)
        distances = list()
        for i in range(n):
            for j in range(i + 1, n):
                u = training_set[i]
                v = training_set[j]

                dist = distance.mahalanobis(u, v, i_cov)

                distances.append(dist)

        return np.mean(distances)




def test():
    X, y = load_breast_cancer(return_X_y=True)
    positive_examples = X[y == 1]
    n_positive_examples = positive_examples.shape[0]
    split = int(0.8*n_positive_examples)
    train, test = positive_examples[:split], positive_examples[split:]
    negative_examples = X[y == 0]
    btpm = BTPM(10, 0.5, 0)
    btpm.fit(train)
    predicted = btpm.predict(negative_examples)
    predicted = btpm.predict(test)
    print(predicted > 0.6)
    #print(f"predited value = {btpm.predict(test)}")

if __name__ == "__main__":
    test()
