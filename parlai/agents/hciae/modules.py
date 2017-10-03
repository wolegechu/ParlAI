import torch
import torch.nn as nn
from torch.autograd import Variable

import math

class HistoryEncoder(nn.Module):
    def __init__(self, opt, dictionary):
        super(HistoryEncoder, self).__init__()
        emb = opt['embedding_size']
        hsz = opt['hidden_size']
        nlayer = opt['rnn_layers']

        self.opt = opt
        self.embedding_size = emb
        self.hidden_size = hsz
        self.num_layers = nlayer

        self.embedding = nn.Embedding(len(dictionary), emb)
        self.h_encoder = nn.LSTM(emb, hsz, nlayer)

    def forward(self, input, lengths):
        # input: [batch_size, memory_length, max_sentence_length]
        # lengths: Variable
        # output:
        #   history: [batch_size, memory_length, hidden_size]

        lengths = lengths.data
        memory_embeddings = []
        for i in range(input.size(0)):
            memory = self.embedding(input[i])
            memory = memory.unsqueeze(0)
            memory_embeddings.append(memory)
        memory_embeddings = torch.cat(memory_embeddings)

        memory_length = memory_embeddings.size(1)
        batch_size = memory_embeddings.size(0)
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        if self.opt['cuda']:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # lengths [batch_size, memory_length]
        s_idx = lengths[0].nonzero()[0][0]
        lengths = lengths[:, s_idx:]
        history = []

        for i in range(memory_length):
            temp = memory_embeddings[:, i]
            data = sort_embeddings(temp, lengths[:, i])
            
            output, _ = self.h_encoder(data, (h0, c0))
            output, o_len = nn.utils.rnn.pad_packed_sequence(output)
            output_embedding = torch.cat([output.transpose(0, 1)[i][o_len[i]-1].unsqueeze(0) for i in range(batch_size)])
            output_embedding = output_embedding.unsqueeze(1)
            history.append(output_embedding)

        history = torch.cat(history, 1)

        return history

def sort_embeddings(embeddings, lens):
    # lens Variable
    lens, perm_idx = lens.sort(0, descending=True)
    embeddings = embeddings[perm_idx]
    data = nn.utils.rnn.pack_padded_sequence(embeddings, lens.tolist(), batch_first=True)
    return data

class QueryEncoder(nn.Module):
    def __init__(self, opt, dictionary):
        super(QueryEncoder, self).__init__()
        emb = opt['embedding_size']
        hsz = opt['hidden_size']
        nlayer = opt['rnn_layers']

        self.opt = opt
        self.num_layers = nlayer
        self.hidden_size = hsz

        self.embedding = nn.Embedding(len(dictionary), emb)
        self.q_encoder = nn.LSTM(emb, hsz, nlayer)

    def forward(self, input, lengths):
        batch_size = input.size(0)
        queries = self.embedding(input)
        data = sort_embeddings(queries, lengths.data)

        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        if self.opt['cuda']:
            h0 = h0.cuda()
            c0 = c0.cuda()

        output, _ = self.q_encoder(data, (h0, c0))
        output, o_len = nn.utils.rnn.pad_packed_sequence(output)
        output_embedding = torch.cat([output.transpose(0, 1)[i][o_len[i]-1].unsqueeze(0) for i in range(batch_size)])

        return output_embedding

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Conv2d(2048, 512, 2, 2)

    def forward(self, input):
        batch_size = input.size(0)
        img = self.conv(input)
        img = img.view(batch_size, 512, -1)
        return img

# class AttM(nn.Module):
#     def __init__(self, t, d):
#         super(att_i, self).__init__()
#         self.t = t
#         self.d = d
#         self.W_i = nn.parameter.Parameter(torch.Tensor(t, d))
#         self.W_c = nn.parameter.Parameter(torch.Tensor(t, d))
#         self.w_a = nn.parameter.Parameter(torch.Tensor(t))
#         self.soft_max = nn.Softmax()
#         self.tanh = nn.Tanh()
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.d)
#         self.W_i.data.uniform_(-stdv, stdv)
#         self.W_c.data.uniform_(-stdv, stdv)
#         self.w_a.data.uniform_(-stdv, stdv)

    
#     def forward(self, M, m):
#         # z_i = w_a.t x tanh(W_i x M_i + W_c x m_c x 1.t)
#         # W_i, W_c t*d
#         # M 8*512(d)*49(t) m 8*512
#         one = Variable(torch.ones(self.t), requires_grad=False)
#         M = torch.transpose(M, 1, 2) # 8*49(t)*512(d)
#         a1 = M.matmul(self.W_i.t())
#         a2 = m.matmul(self.W_c.t()).unsqueeze(2)
#         a2 = a2.matmul(one.unsqueeze(0))
#         z_i = self.tanh(a1 + a2).matmul(self.w_a)
#         alpha = self.soft_max(z_i)
#         alpha = alpha.unsqueeze(2)
#         alpha = alpha.expand_as(M)
#         return alpha * M

class AttM(nn.Module):
    def __init__(self, d):
        super(AttM, self).__init__()
        self.d = d
        self.m_linear = nn.Linear(d, 256)
        self.M_linear = nn.Linear(d, 256)
        self.soft_max = nn.Softmax()
        self.tanh = nn.Tanh()
    
    def forward(self, M, m):
        # z_i = w_a.t x tanh(W_i x M_i + W_c x m_c x 1.t)
        # W_i, W_c t*d
        # M 8*49(t)*512(d) m 8*512
        t = M.size(1)
        M = M.contiguous()
        M = M.view(-1, self.d) # [8*49(t), 512(d)]
        a1 = self.M_linear(M)
        a2 = self.m_linear(m)
        a1 = a1.view(-1, t, 256) # [8, 49, 256]
        a2 = a2.unsqueeze(2) # [8, 256, 1]
        z = torch.bmm(a1, a2).squeeze()

        if z.dim() == 1:
            z = z.unsqueeze(1)

        alpha = self.soft_max(z)
        alpha = alpha.unsqueeze(2)
        M = M.view(-1, t, self.d)
        alpha = alpha.expand_as(M)
        output = alpha * M
        output = output.sum(dim=1)
        return output


class ConcatM(nn.Module):
    def __init__(self, d):
        super(ConcatM, self).__init__()
        self.d = d
        self.W = nn.parameter.Parameter(torch.Tensor(d, 3 * d))
        self.tanh = nn.Tanh()
        
        stdv = 1. / math.sqrt(self.d * 3)
        self.W.data.uniform_(-stdv, stdv)
    
    def forward(self, m_q, m_t, m_i):
        input = torch.cat((m_q, m_t, m_i), 1)
        return self.tanh(input.matmul(self.W.t()))

class HCIAE(nn.Module):
    def __init__(self, opt, dictionary):
        super(HCIAE, self).__init__()
        self.opt = opt

        hsz = opt['hidden_size']
        emb = opt['embedding_size']
        self.answer_embedder = nn.Embedding(len(dictionary), emb)
        self.q_encoder = QueryEncoder(opt, dictionary)
        self.H_encoder = HistoryEncoder(opt, dictionary)
        self.att_image = AttM(hsz)
        self.att_history = AttM(hsz)
        self.i_encoder = ImageEncoder()
        self.qh_linear = nn.Linear(2 * hsz, hsz)
        self.concatM = ConcatM(hsz)

        if opt['cuda']:
            self.q_encoder.cuda()
            self.H_encoder.cuda()
            self.att_image.cuda()
            self.att_history.cuda()
            self.i_encoder.cuda()
    
    def forward(self, histories, queries, history_lengths, query_lengths, images):
        images = self.i_encoder(images)
        #print('image size', images.size())
        queries = self.q_encoder(queries, query_lengths)
        #print('Histories size before encode ', histories.size())
        histories = self.H_encoder(histories, history_lengths)
        #print('Before History Attention - histories size, queries size', histories.size(), queries.size())
        histories = self.att_history(histories, queries)

        query_to_image = torch.cat((histories, queries), 1)
        query_to_image = self.qh_linear(query_to_image)

        #print('Before Image Attention - image size, queries size', images.size(), query_to_image.size())
        images = self.att_image(images.transpose(1, 2), query_to_image)
        output = self.concatM(queries, histories, images)
        return output

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, opt, dictionary):
        super().__init__()
        self.dict = dictionary
        self.h2o = nn.Linear(hidden_size, len(dictionary))
        self.dropout = nn.Dropout(opt['dropout'])
        self.rnn = nn.GRU(input_size, hidden_size, num_layers)

    def hidden_to_idx(self, hidden, dropout=False):
        """Converts hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('Bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        scores = self.h2o(hidden)
        if dropout:
            scores = self.dropout(scores)
        _, idx = scores.max(1)
        return idx, scores

    def forward(self, input, state):
        output, state = self.rnn(input, state)
        return self.hidden_to_idx(output, dropout=self.training)