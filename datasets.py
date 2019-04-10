from torchtext import data
from torchtext import datasets
import torch

def make_reddit(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
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
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS

def make_reddit_with_indices(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  INDEX = data.Field(sequential=False, use_vocab=False, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train.tok.clean.index"+extrasuffix+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None) ,('index',INDEX), ('topics', TOPICS)])
  val = data.TabularDataset(path=base_path+"/valid.tok.clean.index"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  test = data.TabularDataset(path=base_path+"/test.tok.clean.index.loremoved200"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  outdomain_test = data.TabularDataset(path=base_path+"/oodtest.tok.clean.index.loremoved200"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS, INDEX

def make_reddit_baseline(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  outdomain_test = data.TabularDataset(path=base_path+"/oodtest"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS

def make_reddit_baseline_with_indices(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  INDEX = data.Field(sequential=False, use_vocab=False, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  outdomain_test = data.TabularDataset(path=base_path+"/oodtest"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('index',INDEX)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS, INDEX

def make_reddit_ensemble(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None) , ('topics', TOPICS)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('topics', TOPICS)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('topics', TOPICS)])
  outdomain_test = data.TabularDataset(path=base_path+"/oodtest"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None), ('topics', TOPICS)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS


def make_reddit2(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None) , ('topics', TOPICS)])
  outdomain_test = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  val = data.TabularDataset(path=base_path+"/oodtest"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('username', None)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  train_iter, val_iter, test_iter, outdomain_test_iter = data.BucketIterator.splits((train, val, test, outdomain_test), batch_sizes=(batch_size, 256, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter, outdomain_test_iter), TEXT, LABEL, TOPICS

def make_ted(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('topics', TOPICS)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  print (LABEL.vocab.stoi)
  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter), TEXT, LABEL, TOPICS

def make_reddit_gender(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="",domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  train = data.TabularDataset(path=base_path+"/train"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  test = data.TabularDataset(path=base_path+"/test"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  print (LABEL.vocab.stoi)
  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter), TEXT, LABEL, TOPICS

def make_amazon(batch_size, device=-1, vectors=None, base_path="", suffix="",extrasuffix="", domain="", oodname="", topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
  if not topics:
    train = data.TabularDataset(path=base_path+"/"+domain+".train.lower.tok"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  else:
    train = data.TabularDataset(path=base_path+"/"+domain+".train.lower.tok"+suffix+extrasuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('topics', TOPICS)])
  val = data.TabularDataset(path=base_path+"/"+domain+".valid.lower.tok"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  test = data.TabularDataset(path=base_path+"/"+domain+".test.lower.tok"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)])
  oodnames = oodname.split(",")
  outdomain_test = []
  for oodname in oodnames:
    outdomain_test.append(data.TabularDataset(path=base_path+"/"+oodname+".test.lower.tok"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL)]))

  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  all_iters = data.BucketIterator.splits(tuple([train, val, test] + outdomain_test), batch_sizes=tuple([batch_size]*(3+len(outdomain_test))), device=device, repeat=False, sort_key=lambda x: len(x.text))
  # train_iter, val_iter, test_iter, outdomain_test_iters
  return all_iters, TEXT, LABEL, TOPICS

def make_toefl(batch_size, device=-1, vectors=None, base_path="", suffix="", testsuffix=""):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  PROMPT = data.Field(sequential=False, batch_first=True)

  train = data.TabularDataset(path=base_path+"/train"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('takerid', None), ('prompt', PROMPT)])
  val = data.TabularDataset(path=base_path+"/valid"+suffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('takerid', None), ('prompt', PROMPT)])
  test = data.TabularDataset(path=base_path+"/valid"+testsuffix+".txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('takerid', None), ('prompt', PROMPT)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=60000)
  LABEL.build_vocab(train)
  PROMPT.build_vocab(train)
  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter), TEXT, LABEL, PROMPT

def make_demog(batch_size, device=-1, vectors=None, base_path=""):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  PROMPT = data.Field(sequential=False, batch_first=True)

  train = data.TabularDataset(path=base_path+"/train.txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('prompt', PROMPT)])
  val = data.TabularDataset(path=base_path+"/valid.txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('prompt', PROMPT)])
  test = data.TabularDataset(path=base_path+"/test.txt", format="tsv", fields=[('text',TEXT), ('label', LABEL), ('prompt', PROMPT)])
  # train, test = datasets.REDDIT.splits(TEXT, LABEL)
  TEXT.build_vocab(train, vectors=vectors, max_size=60000)
  LABEL.build_vocab(train)
  PROMPT.build_vocab(train)
  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, batch_size, batch_size, batch_size), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter), TEXT, LABEL, PROMPT

dataset_map = {
  "REDDIT": make_reddit,
  "REDDITI": make_reddit_with_indices,
  "REDDIT_ENSEMBLE": make_reddit_ensemble,
  "REDDIT_BASELINE": make_reddit_baseline,
  "REDDIT_BASELINEI": make_reddit_baseline_with_indices,
  "REDDIT_GENDER": make_reddit_gender,
  "REDDIT2": make_reddit2,
  "TED": make_ted,
  "AMAZON": make_amazon,
  "TOEFL": make_toefl,
  "DEMOG": make_demog,
}


if __name__ == '__main__':
  (tr, te), T, L = make_reddit(64)
  print("[REDDIT] vocab: {} labels: {}".format(len(T.vocab), len(L.vocab)))
  print("[REDDIT] train: {} test {}".format(len(tr.dataset), len(te.dataset)))
