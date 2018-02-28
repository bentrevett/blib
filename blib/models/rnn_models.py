import torch
import torch.nn as nn

class RNNClassification(nn.Module):
    """
    Standard bidirectional LSTM
    """
    def __init__(self, input_size, output_dim, embedding_dim=256, hidden_dim=256, rnn_type='LSTM', n_layers=2, bidirectional=True, dropout=0.5):
        super(RNNClassification, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        #set these as attributes as we need in the forward method
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError(f'rnn_type must be LSTM, GRU or RNN! Got {rnn_type}')

        #linear input size is num_directions * hidden_dim
        fc_inp = 2 * hidden_dim if bidirectional else hidden_dim 
        self.fc = nn.Linear(fc_inp, output_dim)

        #layer normalization
        self.ln = LayerNorm(hidden_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        #x = [bsz, seq. len]
        #print(x.shape)

        x = self.embedding(x)

        #x = [bsz, seq. len, emb. dim.]
        #print(x.shape)

        #reshape as need to be [seq. len, bsz, emb. dim] for the rnn
        x = self.do(x.permute(1, 0, 2))

        #x = [seq. len, bsz, emb. dim]
        #print(x.shape)

        if self.rnn_type == 'LSTM':
            o, (h, _) = self.rnn(x)
        else:
            o, h = self.rnn(x)

        #   h is [num dir * num layer, bsz]
        #   the first dim of h goes [layer1 forward, layer1 backward, layer2 forward, layer2 backward]
        #   top layer forward == h[-2,:,:], top layer backward == h[-1,:,:]
        #   so to get the final forward+backward hidden, use h[-2:,:,:]

        #   o is [seq len, bsz, hid_dim * bum dir]
        #   last dim of o is cat(forward, backward, so o[-1,:,:hid dim] is the final forward hidden state, equal to h[-2,:,:]
        #   to get the final backward state, you need to get the first element of the seq. len and the last hid dim of the final dimension
        #   i.e. o[0, :, hid dim:], which equals h[-1,:,:]

        #assert torch.equal(o[0, :, self.hidden_dim:], h[-1, :, :])
        #assert torch.equal(o[-1, :, :self.hidden_dim], h[-2, :, :])

        #h = [n_layers*n_directions, bsz, hid. dim.]
        #print(h.shape)

        if self.bidirectional:
            #h[-2,:,:] = [bsz, hid dim.]
            #h[-1,:,:] = [bsz, hid dim.]
            #h = [bsz, hid dim. * 2]
            h = self.do(torch.cat((h[-2,:,:], h[-1,:,:]), dim=1))
        else:
            #h = [bsz, hid dim.]
            h = self.do(h[-1,:,:])

        output = self.fc(h)

        #output = [bsz, output dim.]

        return output

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    
