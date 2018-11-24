from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Remove topical words')
parser.add_argument('--base_path', type=str, required=True,
                      help='path of base folder')
parser.add_argument('--suffix', type=str, default="",
                      help='suffix like _10, _5, _2 or empty string')
parser.add_argument('--char', action='store_true')
parser.add_argument('--lex', action='store_true')
parser.add_argument('--brown', action='store_true')
parser.add_argument('--output_proba', action='store_true')
parser.add_argument('--predict_ood', action='store_false')
parser.add_argument('--train_prob_path', type=str, help='path of file to write output probabilites for train set')
parser.add_argument('--dev_prob_path', type=str, help='path of file to write output probabilites for valid set')
parser.add_argument('--test_prob_path', type=str, help='path of file to write output probabilites for test set')

args = parser.parse_args()

base_path = args.base_path
ftrain = open(base_path+"/train"+args.suffix+".txt")
fdev = open(base_path+"/valid"+args.suffix+".txt")
ftest = open(base_path+"/test"+args.suffix+".txt")
fouttest = open(base_path+"/oodtest"+args.suffix+".txt")
brownc = pickle.load(open(base_path+"/brown_cluster.pkl","rb"))

texts = []
trainY = []

char_cv = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), max_features=5000)
cv = CountVectorizer(max_features=5000)

for l in ftrain:
    p = l.strip().split("\t")
    texts.append(p[0])
    trainY.append(p[1])

trainfeatures = []
if args.lex:
  trainlexX = cv.fit_transform(texts).todense()
  print ("Found lexical features")
  trainfeatures.append(trainlexX)

if args.char:
  traincharX = char_cv.fit_transform(texts).todense()
  print ("Found char n-gram features")
  trainfeatures.append(traincharX)

if args.brown:
  trainbrownX = []
  for text in texts:
    feat = [0 for i in range(100)]
    words = text.split()
    for word in words:
      if word in brownc:
        feat[brownc[word]-1]+=1.0
    sumfeat = sum(feat)+1e-6
    feat = [f/sumfeat for f in feat]
    trainbrownX.append(feat)
  print ("Found brown cluster features")
  trainbrownX = np.array(trainbrownX)
  trainfeatures.append(trainbrownX)

trainX = np.concatenate(trainfeatures, axis=1)
print ("Train features computed")

###test#####
test_texts = []
testY = []
for l in ftest:
  p = l.strip().split("\t")
  test_texts.append(p[0])
  testY.append(p[1])

testfeatures = []
if args.lex:
  testlexX = cv.transform(test_texts).todense()
  testfeatures.append(testlexX)

if args.char:
  testcharX = char_cv.transform(test_texts).todense()
  testfeatures.append(testcharX)

if args.brown:
  testbrownX = []
  for text in test_texts:
    feat = [0 for i in range(100)]
    words = text.split()
    for word in words:
      if word in brownc:
        feat[brownc[word]-1]+=1.0
    sumfeat = sum(feat)+1e-10
    feat = [f/sumfeat for f in feat]
    testbrownX.append(feat)
  print ("Found brown cluster features for the test set")
  testbrownX = np.array(testbrownX)
  testfeatures.append(testbrownX)

testX = np.concatenate(testfeatures, axis=1)
print ("Test features computed")

###test#####
oodtest_texts = []
oodtestY = []
for l in fouttest:
  p = l.strip().split("\t")
  oodtest_texts.append(p[0])
  oodtestY.append(p[1])

oodtestfeatures = []
if args.lex:
  oodtestlexX = cv.transform(oodtest_texts).todense()
  oodtestfeatures.append(oodtestlexX)

if args.char:
  oodtestcharX = char_cv.transform(oodtest_texts).todense()
  oodtestfeatures.append(oodtestcharX)

if args.brown:
  oodtestbrownX = []
  for text in oodtest_texts:
    feat = [0 for i in range(100)]
    words = text.split()
    for word in words:
      if word in brownc:
        feat[brownc[word]-1]+=1.0
    sumfeat = sum(feat)+1e-10
    feat = [f/sumfeat for f in feat]
    oodtestbrownX.append(feat)
  print ("Found brown cluster features for the oodtest set")
  oodtestbrownX = np.array(oodtestbrownX)
  oodtestfeatures.append(oodtestbrownX)

oodtestX = np.concatenate(oodtestfeatures, axis=1)
print ("Test features computed")

if args.output_proba:
  ###dev#####
  dev_texts = []
  devY = []
  for l in fdev:
    p = l.strip().split("\t")
    dev_texts.append(p[0])
    devY.append(p[1])

  devfeatures = []
  if args.lex:
    devlexX = cv.transform(dev_texts).todense()
    devfeatures.append(devlexX)

  if args.char:
    devcharX = char_cv.transform(dev_texts).todense()
    devfeatures.append(devcharX)

  if args.brown:
    devbrownX = []
    for text in dev_texts:
      feat = [0 for i in range(100)]
      words = text.split()
      for word in words:
        if word in brownc:
          feat[brownc[word]-1]+=1.0
      sumfeat = sum(feat)+1e-10
      feat = [f/sumfeat for f in feat]
      devbrownX.append(feat)
    print ("Found brown cluster features for the dev set")
    devbrownX = np.array(devbrownX)
    devfeatures.append(devbrownX)

  devX = np.concatenate(devfeatures, axis=1)
  print ("Dev features computed")

# test_texts = []
# testoutY = []
# for l in fouttest:
#   p = l.strip().split("\t")
#   test_texts.append(p[0])
#   testoutY.append(p[1])

# # testlexX = cv.transform(test_texts).todense()
# testcharX = char_cv.transform(test_texts).todense()
# testbrownX = []
# for text in test_texts:
#   feat = [0 for i in range(100)]
#   words = text.split()
#   for word in words:
#     if word in brownc:
#       feat[brownc[word]-1]+=1.0
#   sumfeat = sum(feat)+1e-10
#   feat = [f/sumfeat for f in feat]
#   testbrownX.append(feat)
# print ("Found brown cluster features")
# testbrownX = np.array(testbrownX)
# print (testcharX.shape, testbrownX.shape)
# testoutX = np.concatenate([testcharX, testbrownX], axis=1)

###model####
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(trainX, trainY)

predY = lr.predict(testX)

print ("Test Accuracy", (1.0*sum(predY==testY))/len(testY))

if args.output_proba:
  predTrainY = lr.predict(trainX)
  print ("Train Accuracy", (1.0*sum(predTrainY==trainY))/len(trainY))

  probTrainY = lr.predict_proba(trainX)
  f = open(args.train_prob_path,"w")
  for p in probTrainY:
    f.write(" ".join([str(t) for t in p])+"\n")
  f.close()

  probdevY = lr.predict_proba(devX)
  f = open(args.dev_prob_path,"w")
  for p in probdevY:
    f.write(" ".join([str(t) for t in p])+"\n")
  f.close()

  probtestY = lr.predict_proba(testX)
  f = open(args.test_prob_path,"w")
  for p in probtestY:
    f.write(" ".join([str(t) for t in p])+"\n")
  f.close()

if args.predict_ood:
  predY = lr.predict(oodtestX)
  print ("OOD Test Accuracy", (1.0*sum(predY==oodtestY))/len(oodtestY))

