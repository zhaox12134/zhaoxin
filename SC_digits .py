from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN, MeanShift, \
    estimate_bandwidth
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')


def AllAlgorithm(estimator, name, data):
    t0 = time()
    predict = estimator.fit_predict(data)
    # print(predict)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, predict),
             metrics.completeness_score(labels, predict),
             metrics.v_measure_score(labels, predict),
             metrics.adjusted_rand_score(labels, predict),
             metrics.adjusted_mutual_info_score(labels, predict,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, predict,
                                      metric='euclidean',
                                      sample_size=n_samples)))
AllAlgorithm(SpectralClustering(n_clusters=n_digits, affinity="nearest_neighbors"),
              name="SC", data=data)

print(39 * '_' + 'PCA' + 40 * '_')

reduced_data = PCA(n_components=10).fit_transform(data)
AllAlgorithm(SpectralClustering(n_clusters=n_digits, affinity="nearest_neighbors"),
             name="SC",
             data=reduced_data)
print(82 * '_')
