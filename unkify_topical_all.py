import pickle
import sys
import argparse

parser = argparse.ArgumentParser(description='Remove topical words')
parser.add_argument('--remove_file', type=str, required=True,
                      help='File containing words to be removed from each class')
parser.add_argument('--train_file', type=str, required=True,
                        help='Train file')
parser.add_argument('--write_file', type=str, required=True,
                        help='File name to write the unkified text')


args = parser.parse_args()
removewords = pickle.load(open(args.remove_file, "rb"))
rwset = set()

for rw in removewords.values():
  rwset.update(rw)

print (type(removewords))
print (len(rwset))
f = open(args.train_file)
f2 = open(args.write_file, "w")

# print (removewords)
for l in f:
  p = l.strip().split("\t")
  # print (p[1])
  # rwords = removewords[p[1].lower()]
  # print (rwords)
  # input("ok")
  newtext = ""
  for w in p[0].split():
    if w in rwset:
      newtext += "UNK "
    else:
      newtext += w + " "
  p[0] = newtext
  f2.write("\t".join(p)+"\n")

print ("Done", len(removewords))
f2.close()
f.close()
