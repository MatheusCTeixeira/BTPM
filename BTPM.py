import numpy as np
from numpy.linalg import inv
from numpy.random import Generator, MT19937
import pandas as pd
from scipy.spatial import distance
from math import exp
from sklearn.datasets import load_iris

class tp_miner_t:
    def __init__(self, training_set, cov_mat, dist_threshold):
        self._ts = training_set
        self._cm = inv(cov_mat)
        self._dt = dist_threshold
        
        self._typical_objs = list()
        self._supported_objs = set()
        self._trained = False

        
    def train(self, force=False):
        """ Train if not trained or if forced, else return previous evaluation
        """
        if not self._trained or force:
            self.search_typical_objects()
            
        return self._typical_objs

    
    def search_typical_objects(self):
        """ Search for typical objects in training set until all objects become
            a typical object or a supported object.
        """
        while len(self._supported_objs) < len(self._ts):
            i, supported_objects = self.get_most_typical()
            most_tipical_object = self._ts[i]

            # add the objects supported by most typical objects to supported set
            self._typical_objs.append(np.array(most_tipical_object))
            self._supported_objs.update(supported_objects)

            
    def get_most_typical(self):
        """ Eval the typicality of all object and return the most typical. """
        typicality = list()
        
        for i, obj in enumerate(self._ts):
            if i in self._supported_objs: # ignore already supported objects
                continue
            
            closer_objects = self.get_closer_objects(obj)
            tp = sum([exp(-0.5*(d/self._dt)**2) for _, d in closer_objects])
            
            # save object index, tipicality value and supported objects
            typicality.append((tp, i, closer_objects))

        # sort by tipicality
        typicality.sort(key=lambda obj:obj[0], reverse=True)

        # returns the index to the most tipical object and supported objects
        most_typical = typicality[0]
        index = most_typical[1]
        supported_objects = set([i for i, dist in most_typical[2]])
        return index, supported_objects

    
    def get_closer_objects(self, obj):
        """Returns a numpy array whose elements are a tuple of index to training
           set and the correspondent distance.
        """
        closer_objects = list()
        
        for i, similar_obj in enumerate(self._ts):
            if id(obj) == id(similar_obj):
                continue
            
            dist = distance.mahalanobis(obj, similar_obj, self._cm)
            if dist <= self._dt:
                # save index instead of object to save memory and dist to save
                # of another expensive computation.
                closer_objects.append((i, dist))

        return closer_objects



class btpm_t:
    """ Implementation of Bagging for TPMiner """
    
    def __init__(self, training_set, n, sample_prop, q_size = 0, seed=25111996):
        """
           Parameters:
                training_set: Full training set
                n: Quantity of classifiers
                sample_prop: Proportion of training set to sample
                q_size: How many previous classification to consider
                seed: Seed for reproducibility
        """
        self._ts = pd.DataFrame(data=training_set)
        self._n = n
        self._ratio = sample_prop
        self._rd = Generator(MT19937(seed))
        self._q_size = q_size

        self._past_votes = list()
        self._params = list()
        self._trained = False

        
    def train(self, force=False):
        """ Train if not yet trained or if forced """
        if not self._trained or force:
            self.train_inner_classifiers()

            
    def train_inner_classifiers(self):
        """ Train each inner classifier """
        for i in range(self._n):
            training_set = self.bootstrap() # sample from dataset
            cov_mat = self.covariance_matrix(training_set)
            avg_dist = self.average_distance(training_set, cov_mat)
            tp_miner = tp_miner_t(training_set, cov_mat, avg_dist)
            params = tp_miner.train()
            
            self._params.append((params, cov_mat, avg_dist))

            
    def predict(self, obj):
        votes = list()
        for typical_objs, cov_mat, avg_dist in self._params:
            distances = list()
            i_cov = inv(cov_mat)
            for ref_obj in typical_objs:
                dist = distance.mahalanobis(obj, ref_obj, i_cov)
                distances.append(dist)

            min_dist = min(distances)
            vote = exp(-0.5*(min_dist/avg_dist)**2)
            votes.append(vote)

        avg_votes = sum(votes)/len(votes)
        avg_past_votes = self.avg_past_votes()

        self.update_past_votes(avg_votes)

        if self._q_size == 0:
            return avg_votes
        else:
            return (avg_votes + avg_past_votes) / 2

    
    def avg_past_votes(self):
        total = sum(self._past_votes)
        n = len(self._past_votes)
        return total/n if n else 0.0

    
    def update_past_votes(self, new_vote):
        self._past_votes.append(new_vote)

        if len(self._past_votes) >= self._q_size:
            self._past_votes.pop()
            
    def bootstrap(self):
        t_set = self._ts.sample(frac=self._ratio,
                                random_state=self._rd.integers(self._n))
        t_set = np.array(t_set)
        return t_set

    
    def covariance_matrix(self, training_set):
        # observations are in rows and variables in columns (different from
        # default parameters).
        return np.cov(training_set.T)

    
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

        return sum(distances)/len(distances)
        


def test():
    import matplotlib.pyplot as plt 
    
    features, target = load_iris(return_X_y=True)
    features_a, target_a = features[:50], target[:50]
    features_b, target_b = features[50:], target[50:]
    btpm = btpm_t(features_a[:40], 50, 0.15)

    btpm.train()
    test = np.concatenate((features_a[40:], features_b))
    for i, f in enumerate(test):
        print(i, btpm.predict(f) > 0.1)
        

    # x = np.linspace(-2, 2, 80)
    # y = np.linspace(-2, 2, 80)
    # xx, yy = np.meshgrid(x, y)

    # predict = lambda x, y: btpm.predict(np.array([x, y]))
    # predict = np.vectorize(predict)
    # z = predict(xx, yy)
    # cl = plt.contour(xx, yy, z, cmap=plt.cm.Greys, linewidths=0.5, linestyles="dashed")
    # plt.clabel(cl, fontsize=8)
    # plt.scatter(X, Y, alpha=0.3, color="black")
    # plt.axis("off")
    # plt.show()

if __name__ == "__main__":
    test()
