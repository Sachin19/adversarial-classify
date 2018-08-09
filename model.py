import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

import math

RNNS = ['LSTM', 'GRU']

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Encoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True, rnn_type='GRU'):
    super(Encoder, self).__init__()
    self.bidirectional = bidirectional
    assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
    rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
    self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers,
                        dropout=dropout, bidirectional=bidirectional)

  def forward(self, input, hidden=None):
    self.rnn.flatten_parameters()
    return self.rnn(input, hidden)


class Attention(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim):
    super(Attention, self).__init__()
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

    query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
    keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
    energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
    energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination

class Classifier(nn.Module):
  def __init__(self, embedding, encoder, attention, hidden_dim, num_classes, num_topics=50):
    super(Classifier, self).__init__()
    self.embedding = embedding
    self.encoder = encoder
    self.attention = attention
    self.decoder = nn.Linear(hidden_dim, num_classes)
    self.topic_decoder = nn.Sequential(nn.Linear(hidden_dim, num_topics), nn.LogSoftmax())

    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))


  def forward(self, input, alpha=1.0):
    outputs, hidden = self.encoder(self.embedding(input))
    if isinstance(hidden, tuple): # LSTM
      hidden = hidden[1] # take the cell state

    if self.encoder.bidirectional: # need to concat the last 2 hidden layers
      hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    else:
      hidden = hidden[-1]

    # max across T?
    # Other options (work worse on a few tests):
    # linear_combination, _ = torch.max(outputs, 0)
    # linear_combination = torch.mean(outputs, 0)

    energy, linear_combination = self.attention(hidden, outputs, outputs)
    logits = self.decoder(linear_combination)
    reverse_linear_comb = ReverseLayerF.apply(linear_combination, alpha)
    topic_logprobs = self.topic_decoder(reverse_linear_comb)
    return logits, energy, topic_logprobs


class CNN_Text(nn.Module):

  def __init__(self, args, num_topics=50):
      super(CNN_Text, self).__init__()
      self.args = args

      V = args.embed_num
      D = args.embed_dim
      C = args.nlabels
      Ci = 1
      Co = args.kernel_num
      Ks = args.kernel_sizes

      self.embed = nn.Embedding(V, D)
      # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
      self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
      '''
      self.conv13 = nn.Conv2d(Ci, Co, (3, D))
      self.conv14 = nn.Conv2d(Ci, Co, (4, D))
      self.conv15 = nn.Conv2d(Ci, Co, (5, D))
      '''
      self.dropout = nn.Dropout(args.drop)
      self.fc1 = nn.Linear(len(Ks)*Co, C)
      self.topic_decoder = nn.Sequential(nn.Linear(len(Ks)*Co, num_topics), nn.LogSoftmax())

  def conv_and_pool(self, x, conv):
      x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
      x = F.max_pool1d(x, x.size(2)).squeeze(2)
      return x

  def forward(self, x, alpha=1.0):
      x = x.permute(1, 0)
      x = self.embed(x)  # (N, W, D)

      x = Variable(x)

      x = x.unsqueeze(1)  # (N, Ci, W, D)

      x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

      x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

      x = torch.cat(x, 1)

      '''
      x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
      x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
      x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
      x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
      '''
      x = self.dropout(x)  # (N, len(Ks)*Co)
      logit = self.fc1(x)  # (N, C)
      reverse_x = ReverseLayerF.apply(x, alpha)
      topic_logprobs = self.topic_decoder(reverse_x)
      return logit, None, topic_logprobs
