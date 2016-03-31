from diploma_lib import *
from prepare_data import *

def baseline_active(foldname, twenty_train_data, twenty_train_target, twenty_test_data, twenty_test_target):
    #f = open('baseline_active.txt', 'w')
    #baseline active learning solution
    alpha = 20 #initial training set
    betha = int(len(twenty_train_data) / alpha) - 2 #number of iterations
    gamma = 20 #sampling volume

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
    #print 'baseline active learning solutionl'
    diploma_res_print(foldname, len(labeled_train_data), score)
    for t in range(0, betha):
    #while len(unlabeled_train_data) > kol:
        labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_random_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, gamma)
        baseline_active_clf.fit(labeled_train_data, labeled_train_target)
        predicted = baseline_active_clf.predict(twenty_test_data)
        score = f1_score(twenty_test_target, predicted, average='macro')
        diploma_res_print(foldname, len(labeled_train_data), score)

    #f.close()