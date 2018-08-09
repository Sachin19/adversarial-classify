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

f = open(args.train_file)
f2 = open(args.write_file, "w")

for l in f:
  p = l.strip().split("\t")
  rwords = removewords[p[1]]
  newtext = ""
  for w in p[0].split():
    if w in rwords:
      newtext += "UNK "
    else:
      newtext += w + " "
  p[0] = newtext
  f2.write("\t".join(p)+"\n")

f2.close()
f.close()
