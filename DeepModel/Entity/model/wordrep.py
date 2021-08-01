from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np

class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.gpu = data.HP_gpu
        self.batch_size = data.HP_batch_size
        self.input_size = data.word_emb_dim
        self.embedding_dim = data.word_emb_dim
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim).requires_grad_(data.emb_grad)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        self.drop = nn.Dropout(data.HP_dropout)
        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()




    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs):
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent


