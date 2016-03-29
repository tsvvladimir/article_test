from diploma_lib import *
from prepare_data import *

def baseline_active():
    f = open('baseline_active.txt', 'w')
    #baseline active learning solution
    alpha = 10 #initial training set
    betha = 1100 #number of iterations
    gamma = 10 #sampling volume

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
    print 'baseline active learning solution'
    diploma_res_print(f, len(labeled_train_data), score)
    for t in range(1, betha):
        labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target = diploma_random_sampling(labeled_train_data, labeled_train_target, unlabeled_train_data, unlabeled_train_target, gamma)
        baseline_active_clf.fit(labeled_train_data, labeled_train_target)
        predicted = baseline_active_clf.predict(twenty_test_data)
        score = f1_score(twenty_test_target, predicted, average='macro')
        diploma_res_print(f, len(labeled_train_data), score)

    f.close()