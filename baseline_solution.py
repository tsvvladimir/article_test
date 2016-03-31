from diploma_lib import *
from prepare_data import *
import prepare_data

def baseline_solution(foldname):
    '''
    twenty_train_data = pickle.load( open( twenty_train_data, "rb" ) )
    twenty_train_target = pickle.load( open( twenty_train_target, "rb" ) )
    twenty_test_data = pickle.load( open( twenty_test_data, "rb" ) )
    twenty_test_target = pickle.load( open( twenty_test_target, "rb" ) )
    '''

    twenty_train_data = getattr(prepare_data, foldname + '_train_data')
    twenty_train_target = getattr(prepare_data, foldname + '_train_target')
    twenty_test_data = getattr(prepare_data, foldname + '_test_data')
    twenty_test_target = getattr(prepare_data, foldname + '_test_target')

    #f = open('baseline_solution.txt', 'w')
    #baseline solution
    baseline_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])
    #print twenty_train_data[0]
    #print twenty_train_data[1]
    #twenty_train_target = np.array(map(str, twenty_train_target))
    #twenty_train_target = twenty_train_target.tolist()
    #print twenty_train_target[0]
    #print twenty_train_target[1]
    baseline_clf.fit(twenty_train_data, twenty_train_target)
    #print twenty_test_data[0]
    #print twenty_test_data[1]
    predicted = baseline_clf.predict(twenty_test_data)
    score = f1_score(twenty_test_target, predicted, average='macro')
    #print 'baseline solution:'
    diploma_res_print(foldname, len(twenty_train_data), score)
    #f.close()