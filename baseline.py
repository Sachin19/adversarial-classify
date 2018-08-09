from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np

base_path = "/usr1/home/sachink/data/ethics_project/text_chunks/europe_data"
ftrain = open(base_path+"/chunk_5_train.txt")
fdev = open(base_path+"/chunk_5_dev.txt")
ftest = open(base_path+"/chunk_5_test.txt")
fouttest = open(base_path+"/../non_europe_data/chunk_5_test.txt")
brownc = pickle.load(open(base_path+"/brownids.pkl","rb"))

texts = []
trainY = []

char_cv = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), max_features=1000)
cv = CountVectorizer(max_features=1000)

for l in ftrain:
    p = l.strip().split("\t")
    texts.append(p[0])
    trainY.append(p[1])

trainlexX = cv.fit_transform(texts).todense()
print ("Found lexical features")

# traincharX = char_cv.fit_transform(texts).todense()
# print ("Found char n-gram features")

trainbrownX = []
for text in texts:
  feat = [0 for i in range(50)]
  words = text.split()
  for word in words:
    if word in brownc:
      feat[brownc[word]]+=1.0
  sumfeat = sum(feat)
  feat = [f/sumfeat for f in feat]
  trainbrownX.append(feat)
print ("Found brown cluster features")
trainbrownX = np.array(trainbrownX)

trainX = np.concatenate([trainlexX, trainbrownX], axis=1)
print ("Train features computed")

###test#####
test_texts = []
testY = []
for l in ftest:
  p = l.strip().split("\t")
  test_texts.append(p[0])
  testY.append(p[1])

testlexX = cv.transform(test_texts).todense()
# testcharX = char_cv.transform(test_texts).todense()
testbrownX = []
for text in test_texts:
  feat = [0 for i in range(50)]
  words = text.split()
  for word in words:
    if word in brownc:
      feat[brownc[word]]+=1.0
  sumfeat = sum(feat)+1e-10
  feat = [f/sumfeat for f in feat]
  testbrownX.append(feat)
print ("Found brown cluster features")
testbrownX = np.array(testbrownX)
print (testlexX.shape , testbrownX.shape)
testX = np.concatenate([testlexX, testbrownX], axis=1)


test_texts = []
testoutY = []
for l in fouttest:
  p = l.strip().split("\t")
  test_texts.append(p[0])
  testoutY.append(p[1])

testlexX = cv.transform(test_texts).todense()
# testcharX = char_cv.transform(test_texts).todense()
testbrownX = []
for text in test_texts:
  feat = [0 for i in range(50)]
  words = text.split()
  for word in words:
    if word in brownc:
      feat[brownc[word]]+=1.0
  sumfeat = sum(feat)+1e-10
  feat = [f/sumfeat for f in feat]
  testbrownX.append(feat)
print ("Found brown cluster features")
testbrownX = np.array(testbrownX)
print (testlexX.shape, testbrownX.shape)
testoutX = np.concatenate([testlexX, testbrownX], axis=1)

###model####
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(trainX, trainY)

predY = lr.predict(testX)

print ("Test Accuracy", (1.0*sum(predY==testY))/len(testY))

predY = lr.predict(testoutX)

print ("Test Out   Accuracy", (1.0*sum(predY==testoutY))/len(testoutY))

