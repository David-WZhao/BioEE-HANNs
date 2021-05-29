#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Mars
import torch
import util
import json
from HANN import HANN

word_to_idx, word_weight = util.get_word_weight()

hann = HANN(
    word_weight=word_weight,
    word_dim=200,
    n_trigger=20,
    n_role=3,
    dropout=0.5)

# for name, parameters in hann.named_parameters():
#     print(name, ':', parameters.size())

# print('# generator parameters:', sum(param.numel() for param in hann.parameters()))


# Load data

with open("./json/train/PMID-707591.json", "r", encoding="utf-8") as json_file:
    json_str = json.load(json_file)

x_list = [
    torch.LongTensor([word_to_idx[i] for i in json_str["words"][j]])
    for j in range(len(json_str["words"]))
]

edge_index_list = [
    torch.LongTensor(json_str["edge_index"][i]).t().contiguous()
    for i in range(len(json_str["edge_index"]))
]

hypergraph_index = json_str["hypergraph_vertex"]

trigger_index = torch.LongTensor([i[0] for i in json_str["trigger_candidate"]])

trigger_sent = torch.LongTensor([i[1] for i in json_str["trigger_candidate"]])

a1_index = torch.LongTensor([i[0] for i in json_str["event_golden"]])
a2_index = torch.LongTensor([i[1] for i in json_str["event_golden"]])

log_t, log_r = hann(x_list, edge_index_list, hypergraph_index,
                    trigger_index, trigger_sent, a1_index, a2_index)

print(log_t.size(), log_r.size())
