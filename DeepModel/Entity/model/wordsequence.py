from __future__ import print_function
from __future__ import absolute_import
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        self.gpu = data.HP_gpu
        self.lable_alphabet_size = data.label_alphabet_size
        self.batch_size = data.HP_batch_size
        self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.lstm_layer = data.HP_lstm_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        self.tag_dim = data.HP_hidden_dim
        self.hidden2tag = nn.Linear(self.tag_dim, data.label_alphabet_size)
        if self.gpu:
            self.wordrep = self.wordrep.cuda()
            self.lstm = self.lstm.cuda()
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()



    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,data):

        print('wordsequence-------------word_inputs')
        print(word_inputs)
        print('end--------wordsequence-------------word_inputs')
        word_represent = self.wordrep(word_inputs)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        feature_out = self.droplstm(lstm_out.transpose(1,0))
        out = self.hidden2tag(feature_out)
        return out
