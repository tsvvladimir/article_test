from diploma_lib import *
from prepare_data import *

def active_minimum_margin():
    f = open('active_minimum_margin.txt', 'w')
    #baseline active learning solution
    alpha = 100 #initial training set
    betha = 1100 #number of iterations
    gamma = 100 #sampling volume

    #create labeled and unlabeled training set
    labeled_train_data = twenty_train_data[: alpha]
    labeled_train_target = twenty_train_target[: alpha]
    unlabeled_train_data = twenty_train_data[alpha:]
    unlabeled_train_target = twenty_train_target[alpha:]

    baseline_active_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    baseline_active_clf.fit(labeled_train_data, labeled_train_target)
    predicted = baseline_active_clf.predict(twenty_test_data)
    score = f1_score(twenty_test_target, predicted, average='macro')
    print 'baseline active learning solution add with least distance'
    diploma_res_print(f, len(labeled_train_data), score)
    for t in range(1, betha):
        take_idx = []

        #fit classifier on unlabeled data and take gamma with least decision function

        distances = baseline_active_clf.decision_function(unlabeled_train_data)

        #print 'distances shape'
        #print distances.shape[0], distances.shape[1]

        #array with minimum distances for each document
        minimums = []
        for doc in distances:
            minimums.append(np.amin(-np.fabs(np.array(doc))))

        #print len(minimums)

        sorted_idx = np.argsort(minimums)

        #print sorted_idx.shape

        for i in range(0, gamma):
            take_idx.append(sorted_idx[0])
            sorted_idx = np.delete(sorted_idx, 0)

        #print 'start range sampling'
        #print len(unlabeled_train_data)
        labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_range_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, take_idx)
        #print 'start fit'
        baseline_active_clf.fit(labeled_train_data, labeled_train_target)
        #print 'start predict'
        predicted = baseline_active_clf.predict(twenty_test_data)
        #print 'count score'
        score = f1_score(twenty_test_target, predicted, average='macro')
        diploma_res_print(f, len(labeled_train_data), score)
    f.close()