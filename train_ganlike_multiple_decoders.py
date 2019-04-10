from __future__ import print_function

import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
import copy
import codecs
import random

from datasets import dataset_map
from model import *
from optim import Optim

from torchtext.vocab import GloVe

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--data', type=str, default='SST',
                        help='Data corpus: [SST, TREC, IMDB, REDDIT]')
  parser.add_argument('--base_path', type=str, required=True,
                      help='path of base folder')
  parser.add_argument('--suffix', type=str, default="",
                      help='suffix like _10, _5, _2 or empty string')
  parser.add_argument('--extrasuffix', type=str, default="",
                      help='suffix like _10, _5, _2 or empty string')
  parser.add_argument('--rnn_model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
  parser.add_argument('--bottleneck_dim', default=0, type=int,
                    help='Set non zero if add a bottleneck layer')

  parser.add_argument('--save_dir', type=str,
                        help='Directory to save the model')
  parser.add_argument('--model', type=str,
                        help='CNN or RNN or FFN (uses topics as features)')
  parser.add_argument('--model_name', type=str,
                        help='Model name to save')
  parser.add_argument('--topic_loss', type=str, default="kl",
                        help='in [mse|ce|kl]')
  parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
  parser.add_argument('--hidden', type=int, default=128,
                        help='number of hidden units for the RNN encoder')
  parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers of the RNN encoder')
  parser.add_argument('--num_topics', type=int, default=50,
                        help='Number of Topics')
  parser.add_argument('-optim', default='adam',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
  parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
  parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
  parser.add_argument('--param_init', type=float, default=0.1,
                        help='initialize the parameters uniform between [-param_init,param_init]')
  parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')
  parser.add_argument('--pretrain_epochs', type=int, default=2,
                        help='upper epoch limit')
  parser.add_argument('--t_steps', type=int, default=2,
                        help='upper epoch limit')
  parser.add_argument('--c_steps', type=int, default=2,
                        help='upper epoch limit')
  parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use')
  parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
  parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
  parser.add_argument('--topic_drop', type=float, default=0.2,
                        help='dropout')
  parser.add_argument('--bi', action='store_false',
                        help='[DON\'T USE] bidirectional encoder')
  parser.add_argument('--save_output_topics', action='store_true',
                        help='save output topics in file')
  parser.add_argument('--output_topics_save_filename', type=str,
                        help='where to save output topics')
  parser.add_argument('--cuda', action='store_false',
                    help='[DONT] use CUDA')
  parser.add_argument('--load', action='store_true',
                    help='Load and Evaluate the model on test data, dont train')
  parser.add_argument('--latest', action='store_true',
                    help='Load and Evaluate the model on test data, dont train')
  parser.add_argument('--write_attention', action='store_true',
                    help='write attention values to file')

  parser.add_argument('--reset_classifier', action='store_true',help="reset the classifier after every epoch, so that it's not stuck in a specific region and can keep on learning(?_")

  parser.add_argument('--domain', type=str,
                        help='Only for Amazon')
  parser.add_argument('--oodname', type=str,
                        help='Only for Amazon')
  parser.add_argument('--topics', action='store_true',help="whether topics are provided")

  parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
  parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

  return parser

def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  random.seed(seed)
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

def update_stats_topics(accuracy, confusion_matrix, logits, y):
  _, max_ind = torch.max(logits, 1)
  _, max_ind_y = torch.max(y, 1)
  equal = torch.eq(max_ind, max_ind_y)
  correct = int(torch.sum(equal))

  for j, i in zip(max_ind, max_ind_y):
    confusion_matrix[int(i),int(j)]+=1

  return accuracy + correct, confusion_matrix

def pretrain_classifier(model, data, optimizer, criterion, args, epoch):

  model.train()
  accuracy, confusion_matrix = 0.0, np.zeros((args.nlabels, args.nlabels), dtype=int)

  t = time.time()
  total_loss = 0
  num_batches = len(data)

  for batch_num, batch in enumerate(data):

    model.zero_grad()
    x, lens = batch.text
    y = batch.label
    padding_mask = x.ne(1).float()

    logits, energy, topic_logprobs = model(x, padding_mask=padding_mask)
    if energy is not None:
      energy = torch.squeeze(energy)

    loss = criterion(logits.view(-1, args.nlabels), y)

    if torch.isnan(loss):
      print ()
      print ("something has become nan")
      print(logits)
      print (y)
      print (x)
      print (lens)
      input("Press Ctrl+C")
    total_loss += float(loss)

    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r')
    t = time.time()

  print()
  print("[PreTraining Epoch {}: Training Loss]: {:.5f}".format(epoch, total_loss / len(data)), end=" ")
  print("[Training Accuracy]: {}/{} : {:.3f}%".format(epoch, accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  # print(confusion_matrix)
  return total_loss / len(data)

def train_topic_predictor(c_model, t_model, data, optimizer, criterion, args, epoch, steps):

  c_model.train()
  t_model.train()

  accuracy_fromtopics, confusion_matrix_ = 0.0, np.zeros((args.num_topics, args.num_topics), dtype=int)
  t = time.time()
  total_topic_loss = 0
  num_batches = len(data)

  step = 0
  for batch_num, batch in enumerate(data):
    # if step > steps:
    #   break
    c_model.zero_grad()
    t_model.zero_grad()
    x, lens = batch.text
    y = batch.label
    padding_mask = x.ne(1).float()
    logits, energy, textrep = c_model(x, padding_mask=padding_mask)
    topic_logprobs = t_model(textrep)

    topics = batch.topics
    if args.topic_loss == "kl":
      topic_loss = criterion(topic_logprobs, topics)
    elif args.topic_loss == "ce":
      topic_loss = torch.sum(topic_logprobs*topics)
    else:
      g = (topics - torch.exp(topic_logprobs))
      topic_loss = (g*g).sum(dim=-1).mean()

    loss = topic_loss
    total_topic_loss += float(topic_loss)

    accuracy_fromtopics, confusion_matrix_ = update_stats_topics(accuracy_fromtopics, confusion_matrix_, topic_logprobs, topics)
    loss.backward()
    optimizer.step()

    print("[Topic Decoder Epoch {} Batch]: {}/{} in {:.5f} seconds".format(epoch, batch_num, len(data), time.time() - t), end='\r')
    t = time.time()
    step += 1

  print()
  print("[Epoch {} Topic Loss]: {:.5f}".format(epoch, total_topic_loss / len(data)), end=" ")
  print("[Accuracy From Topics]: {}/{} : {}%".format(
        accuracy_fromtopics, np.sum(confusion_matrix_), accuracy_fromtopics / np.sum(confusion_matrix_) * 100))
  # print(confusion_matrix_)
  return total_topic_loss / len(data)

def train_classifier(c_model, t_models, data, optimizer, classify_criterion, topic_criterion, args, epoch, steps):

  c_model.train()
  # print (len(t_models))
  for t_model in t_models:
    t_model.train()
  accuracy, confusion_matrix = 0.0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  accuracy_fromtopics, confusion_matrix_ = 0.0, np.zeros((args.num_topics, args.num_topics), dtype=int)

  t = time.time()
  total_loss = 0
  total_topic_loss = 0
  num_batches = len(data)

  step = 0
  d_id = 0 #which decoder to use. ++ and modulo len(t_models) after every step
  for batch_num, batch in enumerate(data):

    # if step >= steps:
    #   break
    c_model.zero_grad()
    t_models[d_id].zero_grad()

    x, lens = batch.text
    y = batch.label
    padding_mask = x.ne(1).float()

    fake_topics = torch.ones(batch.topics.size()).cuda() #want the model to predict uniform topics
    fake_topics = fake_topics.div(fake_topics.sum(dim=-1, keepdim=True))

    real_topics = batch.topics

    logits, energy, sentrep = c_model(x, padding_mask=padding_mask)
    # print (d_id, t_models[d_id])
    # print (t_models[d_id][1].data.weight.size())
    # input("hello")
    topic_logprobs = t_models[d_id](sentrep).cuda()

    loss = classify_criterion(logits.view(-1, args.nlabels), y)

    if torch.isnan(loss):
      print ()
      print ("something has become nan")
      print(logits)
      print (y)
      print (x)
      print (lens)
      input("Press Ctrl+C")
    total_loss += float(loss)

    if args.topic_loss == "kl":
      topic_loss = topic_criterion(topic_logprobs, fake_topics)
    elif args.topic_loss == "ce":
      topic_loss = -torch.sum(topic_logprobs*fake_topics)
    else:
      g = (fake_topics - torch.exp(topic_logprobs))
      topic_loss = (g*g).sum(dim=-1).mean()

    loss += topic_loss
    total_topic_loss += float(topic_loss)

    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    accuracy_fromtopics, confusion_matrix_ = update_stats_topics(accuracy_fromtopics, confusion_matrix_, topic_logprobs, real_topics)
    loss.backward()
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r')
    t = time.time()
    step += 1
    d_id = (d_id + 1) % len(t_models)

  print()
  print("[Epoch {}: Fake Topic Loss]: {:.5f}".format(epoch, total_topic_loss / len(data)), end=" ")
  print("Loss]: {:.5f}".format(total_loss / len(data)), end=" ")
  print("Training Accuracy]: {}/{} : {}%".format(accuracy, np.sum(confusion_matrix), accuracy / np.sum(confusion_matrix) * 100), end=" ")
  print("accuracy_from_real_topics]: {}/{} : {}%".format(
        accuracy_fromtopics, np.sum(confusion_matrix_), accuracy_fromtopics / np.sum(confusion_matrix_) * 100))
  # print(confusion_matrix)
  return total_loss / len(data)

def evaluate(model, data, criterion, args, datatype='Valid', writetopics=False, itos=None, litos=None):

  model.eval()

  if args.write_attention and itos is not None:
    attention_file = codecs.open(args.save_dir+"/"+datatype+"."+args.model_name+"_attention.txt", "w", encoding="utf8")

  accuracy, confusion_matrix = 0.0, np.zeros((args.nlabels, args.nlabels), dtype=int)

  t = time.time()
  total_loss = 0

  with torch.no_grad():

    for batch_num, batch in enumerate(data):
      x, lens = batch.text
      y = batch.label
      if args.data in ["REDDIT_BASELINEI", "REDDITI"]:
        indices = batch.index.cpu().data.numpy()
        # print (indices.size())
      else:
        indices = np.array(([0]*len(y)))
      padding_mask = x.ne(1).float()

      logits, energy, sentrep = model(x, padding_mask=padding_mask)

      if args.write_attention and itos is not None:
        _, max_ind = torch.max(logits, 1)
        energy = energy.squeeze(1).cpu().data.numpy()
        for sentence, length, attns, ll, mi, index in zip(x.permute(1,0).cpu().data.numpy(), lens.cpu().data.numpy(), energy, y.cpu().data.numpy(), max_ind.cpu().data.numpy(), indices):
          s = ""
          for wordid, attn in zip(sentence[:length], attns[:length]):
            s += str(itos[wordid])+":"+str(attn)+" "
          gold = str(litos[ll])
          pred = str(litos[mi])
          # print (index)
          index = str(index)
          z = s+"\t"+gold+"\t"+pred+"\t"+index+"\n"
          attention_file.write(z)
      bloss = criterion(logits.view(-1, args.nlabels), y)

      if torch.isnan(bloss):
        print ("NANANANANANA")
        print (logits)
        print (y)
        print (x)
        input("Press Ctrl+C")

      total_loss += float(bloss)
      accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)

      print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r')
      t = time.time()

  if args.write_attention and itos is not None:
    attention_file.close()

  print()
  print("[{} loss]: {:.5f}".format(datatype, total_loss / len(data)), end=" ")
  print("[{} accuracy]: {}/{} : {:.3f}%".format(datatype,
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  # print(confusion_matrix)
  return total_loss / len(data)

def main():
  args = make_parser().parse_args()
  print("[Model hyperparams]: {}".format(str(args)))

  cuda = torch.cuda.is_available() and args.cuda
  device = torch.device("cpu") if not cuda else torch.device("cuda:"+str(args.gpu))
  seed_everything(seed=1337, cuda=cuda)
  vectors = None #don't use pretrained vectors
  # vectors = load_pretrained_vectors(args.emsize)

  # Load dataset iterators
  if args.data in ["REDDIT_BASELINEI", "REDDITI"]:
    iters, TEXT, LABEL, TOPICS, INDEX = dataset_map[args.data](args.batch_size, device=device, vectors=vectors, base_path=args.base_path, suffix=args.suffix, extrasuffix=args.extrasuffix, domain=args.domain, oodname=args.oodname, topics=args.topics)
  else:
    iters, TEXT, LABEL, TOPICS = dataset_map[args.data](args.batch_size, device=device, vectors=vectors, base_path=args.base_path, suffix=args.suffix, extrasuffix=args.extrasuffix, domain=args.domain, oodname=args.oodname, topics=args.topics)

  # Some datasets just have the train & test sets, so we just pretend test is valid
  if len(iters) >= 4:
    train_iter = iters[0]
    val_iter = iters[1]
    test_iter = iters[2]
    outdomain_test_iter = list(iters[3:])
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
    classifier_model = CNN_Text_GANLike(args)
    topic_decoder = nn.Sequential(nn.Linear(len(args.kernel_sizes)*args.kernel_num, args.num_topics), nn.LogSoftmax(dim=-1))

  else:
    ntokens, nlabels = len(TEXT.vocab), len(LABEL.vocab)
    args.nlabels = nlabels # hack to not clutter function arguments

    embedding = nn.Embedding(ntokens, args.emsize, padding_idx=1)
    encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers,
                      dropout=args.drop, bidirectional=args.bi, rnn_type=args.rnn_model)

    attention_dim = args.hidden if not args.bi else 2*args.hidden
    attention = BahdanauAttention(attention_dim, attention_dim)

    if args.bottleneck_dim == 0:
      classifier_model = Classifier_GANLike(embedding, encoder, attention, attention_dim, nlabels)
      topic_decoder = [nn.Sequential(nn.Dropout(args.topic_drop), nn.Linear(attention_dim, args.num_topics), nn.LogSoftmax())]
    else:
      classifier_model = Classifier_GANLike_bottleneck(embedding, encoder, attention, attention_dim, nlabels, bottleneck_dim=args.bottleneck_dim)
      topic_decoder = [nn.Sequential(nn.Dropout(args.topic_drop), nn.Linear(args.bottleneck_dim, args.num_topics), nn.LogSoftmax())]

  classifier_model.to(device)
  topic_decoder[0].to(device)

  classify_criterion = nn.CrossEntropyLoss()
  topic_criterion = nn.KLDivLoss(size_average=False)

  classify_optim = Optim(args.optim, args.lr, args.clip)
  topic_optim = Optim(args.optim, args.lr, args.clip)

  for p in classifier_model.parameters():
    if not p.requires_grad:
      print ("OMG", p)
      p.requires_grad = True
    p.data.uniform_(-args.param_init, args.param_init)

  for p in topic_decoder[0].parameters():
    if not p.requires_grad:
      print ("OMG", p)
      p.requires_grad = True
    p.data.uniform_(-args.param_init, args.param_init)

  classify_optim.set_parameters(classifier_model.parameters())
  topic_optim.set_parameters(topic_decoder[0].parameters())

  if args.load:
    if args.latest:
      best_model = torch.load(args.save_dir+"/"+args.model_name+"_latestmodel")
    else:
      best_model = torch.load(args.save_dir+"/"+args.model_name+"_bestmodel")
  else:
    try:
      best_valid_loss = None
      best_model = None

      #pretraining the classifier
      for epoch in range(1, args.pretrain_epochs+1):
        pretrain_classifier(classifier_model, train_iter, classify_optim, classify_criterion, args, epoch)
        loss = evaluate(classifier_model, val_iter, classify_criterion, args)
        #oodLoss = evaluate(classifier_model, outdomain_test_iter[0], classify_criterion, args, datatype="oodtest")

        if not best_valid_loss or loss < best_valid_loss:
          best_valid_loss = loss
          print ("Updating best pretrained_model")
          best_model = copy.deepcopy(classifier_model)
          torch.save(best_model, args.save_dir+"/"+args.model_name+"_pretrained_bestmodel")
        torch.save(classifier_model, args.save_dir+"/"+args.model_name+"_pretrained_latestmodel")

      #alternating training like GANs
      for epoch in range(1, args.epochs + 1):
        for t_step in range(1, args.t_steps+1):
          train_topic_predictor(classifier_model, topic_decoder[-1], train_iter, topic_optim, topic_criterion, args, epoch, args.t_steps)

        if args.reset_classifier:
          for p in classifier_model.parameters():
            if not p.requires_grad:
              print ("OMG", p)
              p.requires_grad = True
            p.data.uniform_(-args.param_init, args.param_init)

        for c_step in range(1, args.c_steps+1):
          train_classifier(classifier_model, topic_decoder, train_iter, classify_optim, classify_criterion, topic_criterion, args, epoch, args.c_steps)
          loss = evaluate(classifier_model, val_iter, classify_criterion, args)
          #oodLoss = evaluate(classifier_model, outdomain_test_iter[0], classify_criterion, args, datatype="oodtest")

        #creating a new instance of a decoder
        attention_dim = args.hidden if not args.bi else 2*args.hidden
        if args.bottleneck_dim == 0:
          topic_decoder.append(nn.Sequential(nn.Dropout(args.topic_drop), nn.Linear(attention_dim, args.num_topics), nn.LogSoftmax()))
        else:
          topic_decoder.append(nn.Sequential(nn.Dropout(args.topic_drop), nn.Linear(args.bottleneck_dim, args.num_topics), nn.LogSoftmax()))

        #attaching a new optimizer to the new topic decode
        topic_decoder[-1].to(device)
        topic_optim = Optim(args.optim, args.lr, args.clip)
        for p in topic_decoder[-1].parameters():
          if not p.requires_grad:
            print ("OMG", p)
            p.requires_grad = True
          p.data.uniform_(-args.param_init, args.param_init)
        topic_optim.set_parameters(topic_decoder[-1].parameters())

        if not best_valid_loss or loss < best_valid_loss:
          best_valid_loss = loss
          print ("Updating best model")
          best_model = copy.deepcopy(classifier_model)
          torch.save(best_model, args.save_dir+"/"+args.model_name+"_bestmodel")
        torch.save(classifier_model, args.save_dir+"/"+args.model_name+"_latestmodel")

    except KeyboardInterrupt:
      print("[Ctrl+C] Training stopped!")

  # if not args.load:
  trainloss = evaluate(best_model, train_iter, classify_criterion, args, datatype='train', writetopics=args.save_output_topics, itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
  valloss = evaluate(best_model, val_iter, classify_criterion, args, datatype='valid', writetopics=args.save_output_topics, itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)

  loss = evaluate(best_model, test_iter, classify_criterion, args, datatype='test', writetopics=args.save_output_topics, itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
  if args.data == "AMAZON":
    oodnames = args.oodname.split(",")
    for oodname, oodtest_iter in zip(oodnames, outdomain_test_iter):
      oodLoss = evaluate(best_model, oodtest_iter, classify_criterion, args, datatype=oodname+"_bestmodel", writetopics=args.save_output_topics)
      oodLoss = evaluate(classifier_model, oodtest_iter, classify_criterion, args, datatype=oodname+"_latest", writetopics=args.save_output_topics)
  # else:
  #   oodLoss = evaluate(best_model, outdomain_test_iter[0], classify_criterion, args, datatype="oodtest_bestmodel", writetopics=args.save_output_topics, itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
  #   oodLoss = evaluate(classifier_model, outdomain_test_iter[0], classify_criterion, args, datatype="oodtest_latest", writetopics=args.save_output_topics)

if __name__ == '__main__':
  main()
