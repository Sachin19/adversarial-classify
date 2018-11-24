from torchtext import data
from torchtext import datasets
import torch

def make_reddit(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix=""):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None) , ('topics', TOPICS)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  outdomain_test = data.TabularDataset(path=base_path+"/oodtest"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, 256, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS

dataset_map = {
  "REDDIT": make_reddit
}


if __name__ == '__main__':
  (tr, te), T, L = make_reddit(64)
  print("[REDDIT] vocab: {} labels: {}".format(len(T.vocab), len(L.vocab)))
  print("[REDDIT] train: {} test {}".format(len(tr.dataset), len(te.dataset)))
