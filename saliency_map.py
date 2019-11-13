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
  parser.add_argument('--testsuffix', type=str, default="",
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
  parser.add_argument('--write_rep', action='store_true',
                    help='write attention values to file')
  parser.add_argument('--saliency_map', action='store_true',
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

def add_noise(text):
    return text

def analyse(model, data, args, datatype="Valid", itos=None, litos=None):
    model.train()

    if args.saliency_map and itos is not None:
        attention_file = codecs.open(args.save_dir+"/"+datatype+"."+args.model_name+"_saliency_map.txt", "w", encoding="utf8") 
    
    if args.write_rep and itos is not None:
        rep_file = codecs.open(args.save_dir+"/"+datatype+"."+args.model_name+"_rep.txt", "w", encoding="utf8") 

    for batch_num, batch in enumerate(data):
        noised_input, lens = add_noise(batch.text)
        y = batch.label
        # print (y.size())
        # print (y.unsqueeze(1).size())
        if args.data in ["REDDIT_BASELINEI", "REDDITI"]:
            indices = batch.index.cpu().data.numpy()
        else:
            indices = np.array(([0]*len(y)))
        padding_mask = noised_input.ne(1).float()
        embeds = model.get_embeddings(noised_input)
        # print(embeds.size())
        # print(padding_mask.size())
        # embeds.requires_grad=True
        logits, energy, sentrep = model.from_embeddings(embeds, padding_mask=padding_mask)
        probs = F.softmax(logits)
        # print(probs.size())
        target_probs = torch.gather(probs, -1, y.unsqueeze(1))
        # print(target_probs.size())
        grad_outputs = torch.zeros_like(probs)
        for i in range(len(y)):
            grad_outputs[i, y[i]] = 1.
        grads = torch.autograd.grad(probs, embeds, grad_outputs)
        # print(len(grads))
        # print(grads[0].size())
        old_saliency_map = grads[0].abs().sum(-1)
        new_saliency_map = (grads[0] * embeds).sum(-1)
        # print(old_saliency_map.size())
        # print(new_saliency_map.size())

        old_saliency_map = old_saliency_map * padding_mask
        new_saliency_map = new_saliency_map * padding_mask
        # print(new_saliency_map.size())
        # input()
        if args.write_rep:
            pass

        if args.saliency_map:
            _, max_ind = torch.max(logits, 1)
            for sentence, saliency, length, ll, mi, index in zip(noised_input.permute(1,0).cpu().data.numpy(), old_saliency_map.permute(1, 0).cpu().data.numpy(), lens.cpu().data.numpy(), y.cpu().data.numpy(), max_ind.cpu().data.numpy(), indices):
              s = ""
              saliency = np.maximum(0., saliency[:length])
              saliency = saliency/np.sum(saliency)
              for wordid, saliency_score in zip(sentence[:length], saliency[:length]):
                s += str(itos[wordid])+":"+str(saliency_score)+" "
              gold = str(litos[ll])
              pred = str(litos[mi])
              # print(s)
              # input()
              # print (index)
              index = str(index)
              z = s+"\t"+gold+"\t"+pred+"\t"+index+"\n"
              attention_file.write(z)
    
    if args.saliency_map and itos is not None:
        attention_file.close()
    
    if args.write_rep and itos is not None:
        rep_file.close()


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
    iters, TEXT, LABEL, INDEX = dataset_map[args.data](args.batch_size, device=device, vectors=vectors, base_path=args.base_path, suffix=args.suffix, extrasuffix=args.extrasuffix, domain=args.domain, oodname=args.oodname, topics=args.topics)
  elif args.data == "TOEFL_TOPICS":
    iters, TEXT, LABEL, PROMPTS, TOPICS = dataset_map[args.data](args.batch_size, device=device, vectors=vectors, base_path=args.base_path, suffix=args.suffix, testsuffix=args.testsuffix, extrasuffix=args.extrasuffix)
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
    topic_decoder = [nn.Sequential(nn.Linear(len(args.kernel_sizes)*args.kernel_num, args.num_topics), nn.LogSoftmax(dim=-1))]

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
      print (topic_decoder)
    else:
      classifier_model = Classifier_GANLike_bottleneck(embedding, encoder, attention, attention_dim, nlabels, bottleneck_dim=args.bottleneck_dim)
      topic_decoder = [nn.Sequential(nn.Dropout(args.topic_drop), nn.Linear(args.bottleneck_dim, args.num_topics), nn.LogSoftmax())]

  classifier_model.to(device)
  topic_decoder[0].to(device)
  # print ("here", topic_decoder[-1])
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

      # print ("here", topic_decoder[-1], topic_decoder)
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
          # print (topic_decoder[-1])
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
#   trainloss = evaluate(best_model, train_iter, classify_criterion, args, datatype='train', writetopics=args.save_output_topics, itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
  valloss = analyse(best_model, val_iter, args, datatype='valid', itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)

  # loss = evaluate(best_model, test_iter, classify_criterion, args, datatype='test', writetopics=args.save_output_topics, itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
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
