from diploma_lib import *
'''
#load data
twenty_train = fetch_20newsgroups(subset='train', remove=('headers'))
twenty_test = fetch_20newsgroups(subset='test', remove=('headers'))

print 'length train:', len(twenty_train.data)
print 'length test:', len(twenty_test.data)

print 'train'
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_train_cats = newsgroups_train.target_names
for cat in newsgroups_train_cats:
    print 'cat:', cat, 'length:', len(fetch_20newsgroups(subset='train', categories=[cat]).data)

print 'test'
newsgroups_test = fetch_20newsgroups(subset='test')
newsgroups_test_cats = newsgroups_test.target_names
for cat in newsgroups_test_cats:
    print 'cat:', cat, 'length:', len(fetch_20newsgroups(subset='test', categories=[cat]).data)

'''
twenty_train = fetch_20newsgroups(subset='train', remove=('headers'))
twenty_test = fetch_20newsgroups(subset='test', remove=('headers'))
#set up train and test datasets
#twenty_train_data = twenty_train.data
#twenty_train_target = twenty_train.target
#twenty_test_data = twenty_test.data
#twenty_test_target = twenty_test.target
pickle.dump(twenty_train.target, open('twenty_train_target.txt', 'wb'))
pickle.dump(twenty_test.target, open('twenty_test_target.txt', 'wb'))
#amount = 2000
twenty_train_data = pickle.load( open( "twenty_train_data.txt", "rb" ) )
twenty_train_target = pickle.load( open( "twenty_train_target.txt", "rb" ) )
twenty_test_data = pickle.load( open( "twenty_test_data.txt", "rb" ) )
twenty_test_target = pickle.load( open( "twenty_test_target.txt", "rb" ) )

#print len(twenty_train_data)
#print len(twenty_test_data)



fold1_train_data = twenty_train_data[2829:11314]
fold1_test_data = twenty_train_data[0:2829]
fold1_train_target = twenty_train_target[2829:11314]
fold1_test_target = twenty_train_target[0:2829]

#print fold1_train_data

fold2_train_data = np.concatenate((twenty_train_data[0:2829], twenty_train_data[5658:11314]), axis=0)
fold2_test_data = twenty_train_data[2829:5658]
fold2_train_target = np.concatenate((twenty_train_target[0:2829], twenty_train_target[5658:11314]), axis=0)
fold2_test_target = twenty_train_target[2829:5658]


fold3_train_data = np.concatenate((twenty_train_data[0:5658], twenty_train_data[8486:11314]), axis=0)
fold3_test_data = twenty_train_data[5658:8486]
fold3_train_target = np.concatenate((twenty_train_target[0:5658], twenty_train_target[8486:11314]), axis=0)
fold3_test_target = twenty_train_target[5658:8486]

fold4_train_data = twenty_train_data[0:8486]
fold4_test_data = twenty_train_data[8486:11314]
fold4_train_target = twenty_train_target[0:8486]
fold4_test_target = twenty_train_target[8486:11314]

#print fold4_test_target

#print type(twenty_test_data)

'''
from sklearn.cross_validation import KFold
kf = KFold(len(twenty_train_data), n_folds=4)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
'''

'''
print datetime.now()

#remove punctuation and stemming
replace_punctuation = string.maketrans(string.punctuation, ' ' * len(string.punctuation)).decode("latin-1")
porter = nltk.PorterStemmer()
for i in range(0, len(twenty_train_data)):
    #print i
    if (i % 100) == 0:
        print i
    text = twenty_train_data[i]
    text = text.lower()
    tokens = word_tokenize(text)
    #text = [word for word in text if word not in stopwords.words('english')]
    text = [porter.stem(t) for t in tokens if t not in stopwords.words('english')]
    twenty_train_data[i] = ' '.join(text)

print datetime.now()

replace_punctuation = string.maketrans(string.punctuation, ' ' * len(string.punctuation)).decode("latin-1")
porter = nltk.PorterStemmer()
for i in range(0, len(twenty_test_data)):
    text = twenty_test_data[i]
    text = text.lower()
    tokens = word_tokenize(text)
    text = [porter.stem(t) for t in tokens if t not in stopwords.words('english')]
    twenty_test_data[i] = ' '.join(text)

print datetime.now()
'''