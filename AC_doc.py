from time import time
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN, MeanShift, \
    estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
vectorizer = TfidfVectorizer(max_df=0.5, max_features=2000, min_df=2, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(dataset.data)

data = X.toarray()

n_samples, n_features = data.shape

n_digits = 20


labels = dataset.target


sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\t\tAMI')

def AllAlgorithm(estimator, name, data):
    t0 = time()
    predict = estimator.fit_predict(data)
    # print(predict)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, predict),
             metrics.completeness_score(labels, predict),
             metrics.v_measure_score(labels, predict),
             metrics.adjusted_rand_score(labels, predict),
             metrics.adjusted_mutual_info_score(labels, predict,
                                                average_method='arithmetic'),
             # metrics.silhouette_score(data, predict,
             #                          metric='euclidean',
             #                          sample_size=n_samples)
             )
          )
    del estimator
AllAlgorithm(AgglomerativeClustering(n_clusters=n_digits, linkage='ward'),
             name="AC", data=data)
bandwidth = estimate_bandwidth(data, quantile=0.3, n_samples=sample_size)
print(39 * '_' + 'PCA' + 40 * '_')

reduced_data = PCA(n_components=10).fit_transform(data)
bandwidth = estimate_bandwidth(data, quantile=0.3, n_samples=sample_size)
AllAlgorithm(AgglomerativeClustering(n_clusters=n_digits, linkage='ward'),
             name="AC",
             data=reduced_data)

print(82 * '_')


