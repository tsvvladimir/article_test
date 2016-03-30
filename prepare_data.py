from diploma_lib import *

#load data
#twenty_train = fetch_20newsgroups(subset='train', remove=('headers'))
#twenty_test = fetch_20newsgroups(subset='test', remove=('headers'))

#set up train and test datasets
#twenty_train_data = twenty_train.data
#twenty_train_target = twenty_train.target
#twenty_test_data = twenty_test.data
#twenty_test_target = twenty_test.target

#amount = 2000
twenty_train_data = pickle.load( open( "twenty_train_data.txt", "rb" ) )
twenty_train_target = pickle.load( open( "twenty_train_target.txt", "rb" ) )
twenty_test_data = pickle.load( open( "twenty_test_data.txt", "rb" ) )
twenty_test_target = pickle.load( open( "twenty_test_target.txt", "rb" ) )



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