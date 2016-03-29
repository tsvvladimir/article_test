from diploma_lib import *
from prepare_data import *

def baseline_solution():
    f = open('baseline_solution.txt', 'w')
    #baseline solution
    baseline_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])
    baseline_clf.fit(twenty_train_data, twenty_train_target)
    predicted = baseline_clf.predict(twenty_test_data)
    score = f1_score(twenty_test_target, predicted, average='macro')
    print 'baseline solution:'
    diploma_res_print(f, len(twenty_train_data), score)
    f.close()