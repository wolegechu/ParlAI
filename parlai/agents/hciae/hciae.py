from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

import torch
from torch import optim
from torch.autograd import Variable

from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter

import os
import copy
import random
import numpy as np

import time

from .modules import HCIAE, Decoder

class HCIAEAgent(Agent):
    """ HCIAEAgent.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_argument_group('HCIAE Arguments')
        arg_group.add_argument('--dropout', type=float, default=.1, help='')
        arg_group.add_argument('--embedding-size', type=int, default=512, help='')
        arg_group.add_argument('--hidden-size', type=int, default=512, help='')
        arg_group.add_argument('--no-cuda', action='store_true', default=False,
                               help='disable GPUs even if available')
        arg_group.add_argument('--gpu', type=int, default=-1,
                               help='which GPU device to use')
        arg_group.add_argument('--rnn-layers', type=int, default=2,
            help='number of hidden layers in RNN decoder for generative output')
        arg_group.add_argument('--optimizer', default='adam',
            help='optimizer type (sgd|adam)')
        arg_group.add_argument('-lr', '--learning-rate', type=float, default=0.01,
                               help='learning rate')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
        if opt['cuda']:
            print('[Using CUDA]')
            torch.cuda.device(opt['gpu'])
        
        if not shared:
            self.opt = opt
            self.id = 'HCIAE'
            self.dict = DictionaryAgent(opt)
            self.answers = [None] * opt['batchsize']

            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))
            self.mem_size = 10
            self.longest_label = 1
            self.writer = SummaryWriter()
            self.writer_idx = 0

            lr = opt['learning_rate']

            self.loss_fn = CrossEntropyLoss()

            self.model = HCIAE(opt, self.dict)
            self.decoder = Decoder(opt['hidden_size'], opt['hidden_size'], opt['rnn_layers'], opt, self.dict)

            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            if opt['optimizer'] == 'sgd':
                self.optimizers = {'hciae': optim.SGD(optim_params, lr=lr)}
                if self.decoder is not None:
                    self.optimizers['decoder'] = optim.SGD(self.decoder.parameters(), lr=lr)
            elif opt['optimizer'] == 'adam':
                self.optimizers = {'hciae': optim.Adam(optim_params, lr=lr)}
                if self.decoder is not None:
                    self.optimizers['decoder'] = optim.Adam(self.decoder.parameters(), lr=lr)
            else:
                raise NotImplementedError('Optimizer not supported.')


            if opt['cuda']:
                self.decoder.cuda()
            
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
        else:
            self.answers = shared['answers']
        
        self.episode_done = True
        self.img_feature = None
        self.last_cands, self.last_cands_list = None, None
    def share(self):
        shared = super().share()
        shared['answers'] = self.answers
        return shared

    def observe(self, observation):
        observation = copy.copy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text'] if self.observation is not None else ''
            prev_dialogue = prev_dialogue + ' __END__ ' + self.observation['labels'][0]
            observation['text'] = prev_dialogue + '\n' + observation['text']
        # else:
        #     self.img_feature = torch.from_numpy(observation['image'].items()[0][1])
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def parse(self, text):
        """Returns:
            query = tensor (vector) of token indices for query
            query_length = length of query
            memory = tensor (matrix) where each row contains token indices for a memory
            memory_lengths = tensor (vector) with lengths of each memory
        """
        sp = text.split('\n')
        query_sentence = sp[-1]
        query = self.dict.txt2vec(query_sentence)
        query = torch.LongTensor(query)
        query_length = torch.LongTensor([len(query)])

        sp = sp[:-1]
        sentences = []
        for s in sp:
            sentences.extend(s.split('\t'))
        if len(sentences) == 0:
            sentences.append(self.dict.null_token)

        num_mems = min(self.mem_size, len(sentences))
        memory_sentences = sentences[-num_mems:]
        memory = [self.dict.txt2vec(s) for s in memory_sentences]
        memory = [torch.LongTensor(m) for m in memory]
        memory_lengths = torch.LongTensor([len(m) for m in memory])
        memory = torch.cat(memory)
        return (query, memory, query_length, memory_lengths)

    def batchify(self, obs):
        """Returns:
            xs = [memories, queries, memory_lengths, query_lengths]
            ys = [labels, label_lengths] (if available, else None)
            (deleted) cands = list of candidates for each example in batch
            valid_inds = list of indices for examples with valid observations
        """
        exs = [ex for ex in obs if ('text' in ex and 'image' in ex)]
        valid_inds = [i for i, ex in enumerate(obs) if ('text' in ex and 'image' in ex)]
        if not exs:
            return [None] * 4
        images = torch.cat([torch.from_numpy(exs[i]['image']['x']).unsqueeze(0) for i in valid_inds])
        parsed = [self.parse(exs[i]['text']) for i in valid_inds]
        queries = torch.cat([x[0] for x in parsed])
        memories = torch.cat([x[1] for x in parsed])
        query_lengths = torch.cat([x[2] for x in parsed])
        memory_lengths = torch.LongTensor(len(exs), self.mem_size).zero_()
        for i in range(len(exs)):
            if len(parsed[i][3]) > 0:
                memory_lengths[i, -len(parsed[i][3]):] = parsed[i][3]

        # bachify memories (batchsize * memory_length * max_sentence_length)
        batch_size = len(valid_inds)
        start_idx = memory_lengths.numpy()[0].nonzero()[0][0]
        idx = 0
        memories_tensor = []
        max_len = torch.max(memory_lengths)
        for i in range(batch_size):
            memory = []
            for j in range(start_idx, 10):
                temp = []
                length = memory_lengths[i][j]
                temp = [memories[idx + i] for i in range(length)]
                temp.extend([0] * (max_len - length))
                idx += length
                memory.append(temp)
            memories_tensor.append(memory)
        memories_tensor = torch.from_numpy(np.array(memories_tensor))

        # bachify queries (batch_size * max_query_length)
        idx = 0
        max_length = max(query_lengths)

        queries_tensor = []
        for i in range(batch_size):
            temp = []
            length = query_lengths[i]
            temp = [queries[idx+i] for i in range(length)]
            temp.extend([0] * (max_length - length))
            idx += length
            queries_tensor.append(temp)
        queries_tensor = torch.from_numpy(np.array(queries_tensor))
        xs = [memories_tensor, queries_tensor, memory_lengths, query_lengths, images]
        ys = None
        self.labels = [random.choice(ex['labels']) for ex in exs if 'labels' in ex]
        if len(self.labels) == len(exs):
            parsed = [self.dict.txt2vec(l) for l in self.labels]
            parsed = [torch.LongTensor(p) for p in parsed]
            label_lengths = torch.LongTensor([len(p) for p in parsed]).unsqueeze(1)
            self.longest_label = max(self.longest_label, label_lengths.max())
            padded = [torch.cat((p, torch.LongTensor(self.longest_label - len(p))
                        .fill_(self.END_TENSOR[0]))) for p in parsed]
            labels = torch.stack(padded)
            ys = [labels, label_lengths]

        return xs, ys, valid_inds

    def predict(self, xs, ys=None):
        is_training = ys is not None
        self.model.train(mode=is_training)
        inputs = [Variable(x) for x in xs]
        output = self.model(*inputs)

        self.decoder.train(mode=is_training)

        output_lines, loss = self.decode(output, ys)
        predictions = self.generated_predictions(output_lines)

        if is_training:
            for o in self.optimizers.values():
                o.zero_grad()
            loss.backward()
            for o in self.optimizers.values():
                o.step()
            self.writer_idx += 1
            #print('Loss: ', loss.data[0])

            #if self.writer_idx == 1:
            #    self.writer.add_graph(self.model, output)

            self.writer.add_histogram('loss', loss.data[0], self.writer_idx)
            self.writer.add_embedding(output.data, tag='output', global_step=self.writer_idx)
            if random.random() < 0.25:
                label = self.dict.vec2txt(ys[0][0].tolist())
                self.writer.add_text('prediction - label', ' '.join(predictions[0]) + ' --- ' + label, self.writer_idx)

        return predictions
    

    def decode(self, output_embeddings, ys=None):
        # output_embedding [batich_size, hidden_size[
        batchsize = output_embeddings.size(0)
        hn = output_embeddings.unsqueeze(0).expand(
            self.opt['rnn_layers'], batchsize, output_embeddings.size(1))
        x = self.model.answer_embedder(Variable(self.START_TENSOR))
        xes = x.unsqueeze(1).expand(x.size(0), batchsize, x.size(1))

        loss = 0
        output_lines =[[] for _ in range(batchsize)]
        done = [False for _ in range(batchsize)]
        total_done = 0
        idx = 0

        while (total_done < batchsize) and idx < self.longest_label:
            # keep producing tokens until we hit END or max length for each ex
            if self.opt['cuda']:
                xes = xes.cuda()
                hn = hn.contiguous()
            #print('Before Decoder size - xes, hn', xes.size(), hn.size())
            preds, scores = self.decoder(xes, hn)
            if ys is not None:
                y = Variable(ys[0][:, idx])
                temp_y = y.cuda() if self.opt['cuda'] else y
                loss += self.loss_fn(scores, temp_y)
            else:
                y = preds
            # use the true token as the next input for better training
            xes = self.model.answer_embedder(y).unsqueeze(0)

            for b in  range(batchsize):
                if not done[b]:
                    token = self.dict.vec2txt([preds.data[b]])
                    if token == self.END:
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)
            idx += 1

        return output_lines, loss
    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]
        xs, ys, valid_inds = self.batchify(observations)

        if xs is None or len(xs[1]) == 0:
            return batch_reply

        # Either train or predict
        predictions = self.predict(xs, ys)

        for i in range(len(valid_inds)):
            #self.answers[valid_inds[i]] = predictions[i][0]
            batch_reply[valid_inds[i]]['text'] = predictions[i][0]
            #batch_reply[valid_inds[i]]['text_candidates'] = predictions[i]
        return batch_reply

    def generated_predictions(self, output_lines):
        return [[' '.join(c for c in o if c != self.END
                        and c != self.dict.null_token)] for o in output_lines]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path
        now_time = time.strftime('%m-%d %H:%M', time.localtime(time.time()))
        path = path + now_time + '.model'
        if path:
            checkpoint = {}
            checkpoint['hciae'] = self.model.state_dict()
            checkpoint['hciae_optim'] = self.optimizers['memnn'].state_dict()
            if self.decoder is not None:
                checkpoint['decoder'] = self.decoder.state_dict()
                checkpoint['decoder_optim'] = self.optimizers['decoder'].state_dict()
                checkpoint['longest_label'] = self.longest_label
            with open(path, 'wb') as write:
                torch.save(checkpoint, write)
