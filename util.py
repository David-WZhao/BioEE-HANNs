#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2020/3/12 16:45
# @Author: Mars
import torch
import gensim
import datetime


def get_word_weight():

    print("Load vocab...")
    start_vocab = datetime.datetime.now()
    word_list = []
    with open("./w2c/all_word.txt", "r", encoding="utf-8") as txt_file:
        all_word = txt_file.readlines()
    for word in all_word:
        word_list.append(word.strip())

    vocab = set(word_list)  
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx["<unk>"] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = "<unk>"
    end_vocab = datetime.datetime.now()
    print("Loading completed! Running time:",
          (end_vocab - start_vocab).seconds, "s.")

    print("Load pre-trained W2C...")
    start_w2c = datetime.datetime.now()
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
        r".\w2c\wikipedia-pubmed-and-PMC-w2v.bin",
        binary=True,
        encoding="UTF-8")
    end_w2c = datetime.datetime.now()
    print("Loading completed! Running time:",
          (end_w2c - start_w2c).seconds, "s.")

    vocab_size = len(word_to_idx)
    embed_size = 200
    word_weight = torch.zeros(vocab_size, embed_size)

    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except BaseException:
            continue
        word_weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

    return word_to_idx, word_weight
