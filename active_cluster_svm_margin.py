#clusterize in 20 clusters and choose cluster center for init learning

from diploma_lib import *
from prepare_data import *
from sklearn.cluster import AgglomerativeClustering
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics import pairwise_distances_argmin_min

def active_cluster_svm_margin():
    #baseline active learning solution
    alpha = 20 #initial training set
    betha = 600 #number of iterations
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
    print 'active cluster svm margin solution'
    diploma_res_print(len(labeled_train_data), score)
    for t in range(1, betha):
        #to do use labeled dataset to train sigmoid

        #f1 for labeled set
        #pred_lab = baseline_active_clf.predict(labeled_train_data)
        #print 'f1 score for labeled:', f1_score(labeled_train_target, pred_lab, average='macro')


        scores = baseline_active_clf.decision_function(unlabeled_train_data)

        #count p1 p2 p3 p4

        def count_p(arr):
            p1 = arr.min()
            p4 = arr.max()
            sorted_arr = sorted(arr)
            a1 = [i for i in sorted_arr if i < 0]
            a2 = [i for i in sorted_arr if i > 0]
            p2 = -100500
            p3 = +100500
            if len(a1) > 0:
                p2 = max(a1)
            if len(a2) > 0:
                p3 = min(a2)
            return [p1, p2, p3, p4]

        #prom_arr = []

        norm_scores = LA.norm(scores)
        n_scores = np.divide(scores, norm_scores)

        '''
        plus_norm = 0
        min_norm = 0
        for line in scores:
            for elem in line:
                if (elem > 0):
                    plus_norm += elem ** 2
                else:
                    min_norm += elem ** 2
        plus_norm = math.sqrt(plus_norm)
        min_norm = math.sqrt(min_norm)
        n_scores = np.array(scores)
        for i in range(0, len(n_scores)):
            for j in range(0, len(n_scores[i])):
                if (n_scores[i][j] > 0):
                    n_scores[i][j] = n_scores[i][j] / plus_norm
                else:
                    n_scores[i][j] = n_scores[i][j] / min_norm
        '''

        #print n_scores
        prom_arr = []
        for lin in range(0, len(n_scores)):
            prom_arr.append(count_p(n_scores[lin]))

        t_prom_arr = np.transpose(np.array(prom_arr))
        #print t_prom_arr
        p1 = np.amin(t_prom_arr[0])
        p2 = np.amax(t_prom_arr[1])
        p3 = np.amin(t_prom_arr[2])
        p4 = np.amax(t_prom_arr[3])
        print 'p1:', p1, 'p2:', p2, 'p3:', p3, 'p4:', p4


        prob = np.divide(1, np.add(1, np.exp(np.multiply(np.array(scores), -1))))
        print 'min proba:', np.amin(prob), 'max proba:', np.amax(prob)

        prob = np.divide(1, np.add(1, np.exp(np.multiply(np.array(n_scores), -1))))
        print 'norm matrix min proba:', np.amin(prob), 'norm matrix max proba:', np.amax(prob)

        doc_score = {}
        for i in range(0, len(unlabeled_train_data)):
            last_elems = (sorted(scores[i]))[-2:]
            doc_score[i] = np.abs(last_elems[0] - last_elems[1])

        sorted_doc_score = sorted(doc_score.items(), key=operator.itemgetter(1))


        #print 'sorted doc score minimum active cluster svm margin', sorted_doc_score[0]

        sample_numbers = np.array([])
        for i in range(0, gamma):
            sample_numbers = np.append(sample_numbers, sorted_doc_score[i][0])

        labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_range_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, sample_numbers)
        baseline_active_clf.fit(labeled_train_data, labeled_train_target)
        predicted = baseline_active_clf.predict(twenty_test_data)
        score = f1_score(twenty_test_target, predicted, average='macro')
        diploma_res_print(len(labeled_train_data), score)