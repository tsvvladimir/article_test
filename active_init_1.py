#clusterize in 20 clusters and choose cluster center for init learning

from diploma_lib import *
from prepare_data import *
from sklearn.cluster import AgglomerativeClustering

def active_init_1():
    #baseline active learning solution
    alpha = 20 #initial training set
    betha = 100 #number of iterations
    gamma = 10 #sampling volume

    tfidf_transformer = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer())
    ])

    #try to implement silhouette analysis for number of clusters
    cluster = KMeans(n_clusters=int(math.sqrt(len(twenty_train_data)/2)))

    unlabeled_train_data = twenty_train_data
    unlabeled_train_target = twenty_train_target

    print 'start transforming'
    unlabeled_matrix = tfidf_transformer.fit_transform(unlabeled_train_data)

    print 'start fitting'
    cluster_distance = cluster.fit_transform(unlabeled_matrix, unlabeled_train_target)

    print 'cluster distance shape'
    print cluster_distance.shape

    init_idx = []
    doc_cluster = []

    for doc in cluster_distance:
        doc_cluster.append(np.argmax(doc))
        print doc_cluster

    #create labeled and unlabeled training set
    #labeled_train_data = twenty_train_data[: alpha]
    #labeled_train_target = twenty_train_target[: alpha]
    #unlabeled_train_data = twenty_train_data[alpha:]
    #unlabeled_train_target = twenty_train_target[alpha:]

    baseline_active_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    baseline_active_clf.fit(labeled_train_data, labeled_train_target)
    predicted = baseline_active_clf.predict(twenty_test_data)
    score = f1_score(twenty_test_target, predicted, average='macro')
    print 'baseline active learning solution'
    diploma_res_print(len(labeled_train_data), score)
    for t in range(1, betha):
        labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_random_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, gamma)
        baseline_active_clf.fit(labeled_train_data, labeled_train_target)
        predicted = baseline_active_clf.predict(twenty_test_data)
        score = f1_score(twenty_test_target, predicted, average='macro')
        diploma_res_print(len(labeled_train_data), score)