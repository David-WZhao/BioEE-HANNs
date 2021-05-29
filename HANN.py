#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Mars
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class HANN(nn.Module):

    def __init__(self, word_weight, word_dim, n_trigger, n_role, dropout):
	
        super(HANN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_weight)
        self.embedding.weight.requires_grad = True

        self.pos_encoder = PositionalEncoding(word_dim, dropout)

        self.dep_gcn = DepGCN(word_dim, word_dim, dropout)

        self.ffnn1 = nn.Linear(word_dim * 2, word_dim)
        self.ffnn2 = nn.Linear(word_dim * 2, n_trigger)
        self.ffnn3 = nn.Linear(word_dim * 2, n_role)

    def depgcn(self, x_list, edge_index_list):

        out_list = []
        sent_list = []

        for x, edge_index in zip(x_list, edge_index_list):
            emb = self.embedding(x)
            emb = emb.unsqueeze(0)
            emb = self.pos_encoder(emb)
            emb = emb.squeeze()
            out, sent = self.dep_gcn(emb, edge_index)
            out_list.append(out)
            sent_list.append(sent)

        outs = torch.cat(out_list, 0)
        sents = torch.cat(sent_list, 0)

        return outs, sents

    def aggregation(self, outs, sents, hyperedge_index):

        for vertex, hyperedge in hyperedge_index:

            e_l = []
            for e in hyperedge:
                e_l.append(sents[e].unsqueeze(0))

            temp = torch.cat((outs[int(vertex)].unsqueeze(0), torch.max(
                torch.cat(e_l, 0).t().contiguous(), 1)[0].unsqueeze(0)), 1)

            outs[int(vertex)] = self.ffnn1(temp).squeeze(0)

        return outs

    def forward(self, x_list, edge_index_list, hyperedge_index, trigger_index,
                trigger_sent, a1_index, a2_index):

        outs, sents = self.depgcn(x_list, edge_index_list)

        outs = self.aggregation(outs, sents, hyperedge_index)

        t_x = outs[trigger_index]
        t_s = sents[trigger_sent]
        t_c = torch.cat((t_x, t_s), 1)
        prob_t = F.log_softmax(self.ffnn2(t_c), dim=1)

        a1 = outs[a1_index]
        a2 = outs[a2_index]

        a_c = torch.cat((a1, a2), 1)

        prob_r = F.log_softmax(self.ffnn3(a_c), dim=1)

        return prob_t, prob_r


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DepGCN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(DepGCN, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = GCNConv(d_model, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, d_model)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # Max-Pooling
        p = torch.max(x.t().contiguous(), 1)[0]
        p = p.unsqueeze(0)
        return x, p