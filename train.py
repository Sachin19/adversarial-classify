import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
import copy

from datasets import dataset_map
from model import *
from torchtext.vocab import GloVe

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--data', type=str, default='SST',
                        help='Data corpus: [SST, TREC, IMDB, REDDIT]')
  parser.add_argument('--rnn_model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
  parser.add_argument('--save_dir', type=str,
                        help='Directory to save the model')
  parser.add_argument('--model', type=str,
                        help='CNN or RNN')
  parser.add_argument('--model_name', type=str,
                        help='Model name to save')
  parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
  parser.add_argument('--hidden', type=int, default=500,
                        help='number of hidden units for the RNN encoder')
  parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers of the RNN encoder')
  parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
  parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
  parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
  parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
  parser.add_argument('--cuda', action='store_false',
                    help='[DONT] use CUDA')
  parser.add_argument('--load', action='store_true',
                    help='Load and Evaluate the model on test data, dont train')
  parser.add_argument('--demote_topics', action='store_true',
                    help='[Demote] topics adversarially while training')
  parser.add_argument('--fine', action='store_true',
                    help='use fine grained labels in SST')

  parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
  parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

  return parser


def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)


def update_stats(accuracy, confusion_matrix, logits, y):
  _, max_ind = torch.max(logits, 1)
  equal = torch.eq(max_ind, y)
  correct = int(torch.sum(equal))

  for j, i in zip(max_ind, y):
    confusion_matrix[int(i),int(j)]+=1

  return accuracy + correct, confusion_matrix


def train(model, data, optimizer, criterion, args):
  model.train()
  accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  t = time.time()
  total_loss = 0
  total_topic_loss = 0
  for batch_num, batch in enumerate(data):
    model.zero_grad()
    # print (batch.fields)
    # input()
    x, lens = batch.text
    y = batch.label
    topics = batch.topics
    # print (topics.size())
    logits, _, topic_logprobs = model(x)
    loss = criterion(logits.view(-1, args.nlabels), y)
    total_loss += float(loss)

    if args.demote_topics:
      topic_loss = -(topic_logprobs * topics).sum(dim=-1).mean()
      loss += topic_loss
      total_topic_loss += float(topic_loss)

    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r', flush=True)
    t = time.time()

  print()
  print("[Topic Loss]: {:.5f}".format(total_topic_loss / len(data)))
  print("[Loss]: {:.5f}".format(total_loss / len(data)))
  print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  print(confusion_matrix)
  return total_loss / len(data)


def evaluate(model, data, optimizer, criterion, args, type='Valid'):
  model.eval()
  accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  t = time.time()
  total_loss = 0
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      # print (len(batch.text))
      x, lens = batch.text
      y = batch.label

      logits, _, topic_logits = model(x)
      total_loss += float(criterion(logits.view(-1, args.nlabels), y))
      accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
      print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r', flush=True)
      t = time.time()

  print()
  print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
  print("[{} accuracy]: {}/{} : {:.3f}%".format(type,
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  print(confusion_matrix)
  return total_loss / len(data)

pretrained_GloVe_sizes = [50, 100, 200, 300]

def load_pretrained_vectors(dim):
  if dim in pretrained_GloVe_sizes:
    # Check torchtext.datasets.vocab line #383
    # for other pretrained vectors. 6B used here
    # for simplicity
    name = 'glove.{}.{}d'.format('6B', str(dim))
    return name
  return None

def main():
  args = make_parser().parse_args()
  print("[Model hyperparams]: {}".format(str(args)))

  cuda = torch.cuda.is_available() and args.cuda
  device = torch.device("cpu") if not cuda else torch.device("cuda:0")
  seed_everything(seed=1337, cuda=cuda)
  vectors = None #don't use pretrained vectors
  # vectors = load_pretrained_vectors(args.emsize)

  # Load dataset iterators
  iters, TEXT, LABEL, TOPICS = dataset_map[args.data](args.batch_size, device=device, vectors=vectors)

  # Some datasets just have the train & test sets, so we just pretend test is valid
  if len(iters) == 4:
    train_iter, val_iter, test_iter, outdomain_test_iter = iters
  elif len(iters) == 3:
    train_iter, val_iter, test_iter = iters
  else:
    train_iter, test_iter = iters
    val_iter = test_iter

  print("[Corpus]: train: {}, test: {}, vocab: {}, labels: {}".format(
            len(train_iter.dataset), len(test_iter.dataset), len(TEXT.vocab), len(LABEL.vocab)))

  if args.model == "CNN":
    args.embed_num = len(TEXT.vocab)
    args.nlabels = len(LABEL.vocab)
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.embed_dim = args.emsize

    model = CNN_Text(args)
  else:
    ntokens, nlabels = len(TEXT.vocab), len(LABEL.vocab)
    args.nlabels = nlabels # hack to not clutter function arguments

    embedding = nn.Embedding(ntokens, args.emsize, padding_idx=1, max_norm=1)
    if vectors: embedding.weight.data.copy_(TEXT.vocab.vectors)
    encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers,
                      dropout=args.drop, bidirectional=args.bi, rnn_type=args.rnn_model)

    attention_dim = args.hidden if not args.bi else 2*args.hidden
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = Classifier(embedding, encoder, attention, attention_dim, nlabels)
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)

  if args.load:
    best_model = torch.load(args.save_dir+"/"+args.model_name+"_bestmodel")
  else:
    try:
      best_valid_loss = None
      best_model = None
      for epoch in range(1, args.epochs + 1):
        train(model, train_iter, optimizer, criterion, args)
        loss = evaluate(model, val_iter, optimizer, criterion, args)

        if not best_valid_loss or loss < best_valid_loss:
          best_valid_loss = loss
          print ("Updating best model")
          best_model = copy.deepcopy(model)
          torch.save(best_model, args.save_dir+"/"+args.model_name+"_bestmodel")


    except KeyboardInterrupt:
      print("[Ctrl+C] Training stopped!")
  loss = evaluate(best_model, test_iter, optimizer, criterion, args, type='Test')
  odLoss = evaluate(best_model, outdomain_test_iter, optimizer, criterion, args, type="OutofDomainTest")

if __name__ == '__main__':
  main()
