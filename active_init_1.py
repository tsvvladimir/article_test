#clusterize in 20 clusters and choose cluster center for init learning

from diploma_lib import *
from prepare_data import *
from sklearn.cluster import AgglomerativeClustering
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics import pairwise_distances_argmin_min

def active_init_1():
    #baseline active learning solution
    alpha = 20 #initial training set
    betha = 100 #number of iterations
    gamma = 20 #sampling volume

    tfidf_transformer = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer())
    ])

    #try to implement silhouette analysis for number of clusters
    #cluster = AgglomerativeClustering(n_clusters=20,affinity='cosine', linkage='complete')
    cluster = KMeans(n_clusters=20)

    unlabeled_train_data = twenty_train_data
    unlabeled_train_target = twenty_train_target

    print 'start transforming'
    unlabeled_matrix = tfidf_transformer.fit_transform(unlabeled_train_data)

    print 'start fitting'
    print datetime.now()
    res = cluster.fit_predict(unlabeled_matrix)
    print datetime.now()

    print 'clustering result'
    print OrderedDict(Counter(res))
    print res.shape

    closest, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, unlabeled_matrix, metric='cosine')

    print closest

    '''
    results = defaultdict(list)
    for idx, val in enumerate(res):
        results[val].append(idx)

    take_idx = []
    for cluster_num in range(0, 20):
        idxset = results[cluster_num]
    '''



    #create labeled and unlabeled training set
    #labeled_train_data = twenty_train_data[: alpha]
    #labeled_train_target = twenty_train_target[: alpha]
    #unlabeled_train_data = twenty_train_data[alpha:]
    #unlabeled_train_target = twenty_train_target[alpha:]
    labeled_train_data = []
    labeled_train_target = []
    labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_range_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, closest)
    print labeled_train_data.shape
    baseline_active_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    baseline_active_clf.fit(labeled_train_data, labeled_train_target)
    predicted = baseline_active_clf.predict(twenty_test_data)
    score = f1_score(twenty_test_target, predicted, average='macro')
    print 'baseline active clustering solution'
    diploma_res_print(len(labeled_train_data), score)
    for t in range(1, betha):
        unlabeled_matrix = tfidf_transformer.fit_transform(unlabeled_train_data)
        print datetime.now()
        res = cluster.fit_predict(unlabeled_matrix)
        print datetime.now()
        closest, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, unlabeled_matrix, metric='cosine')
        print closest
        labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_range_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, closest)
        baseline_active_clf.fit(labeled_train_data, labeled_train_target)
        predicted = baseline_active_clf.predict(twenty_test_data)
        score = f1_score(twenty_test_target, predicted, average='macro')
        diploma_res_print(len(labeled_train_data), score)