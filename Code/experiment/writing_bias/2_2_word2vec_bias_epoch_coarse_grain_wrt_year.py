#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 2_2_word2vec_bias_epoch_coarse_grain_wrt_year.py
# @Author: Hua Zhu
# @Date  : 2022/4/24
# @Desc  :
import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import util_draw
from util_pojo import DrawingData
import util_corpus


def load_embed_model_by_mediaName(media_name, model_type, random_seed_pretrain=None, random_seed_finetuned=None, epoch=None, seg_id=None, year_set=None):
    """
    加载指定media的词向量模型 (未归一化)
    :param media_name:
    :param model_type: 'pretrain' 'finetun' 'alone'
    :param random_seed_pretrain:
    :param random_seed_finetuned:
    :param epoch_k:
    :return:
    """
    print('start load model: ', media_name)

    # (1)
    if seg_id is not None:
        print('epoch=', epoch, ' seg_id=', seg_id)
        if model_type == 'pretrain':
            root_dir = '/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
        elif model_type == 'finetune':
            root_dir = '/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/model_epoch_fine-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + '['+str(epoch)+','+str(seg_id)+']' + '.model'))
        else:
            print('error: Model does not exist')
    # (2)
    if year_set is not None:
        print('epoch=', epoch, ' year_set=', year_set)
        year_set_info = str(year_set[0]) + '_' + str(year_set[-1])
        if model_type == 'pretrain':
            root_dir = '/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
        elif model_type == 'finetune':
            root_dir = '/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/model_epoch_coarse-grain_wrt_year/' + media_name + '/' + year_set_info
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + str(epoch) + '.model'))
        else:
            print('error: Model does not exist')
    # (3)
    if seg_id is None and year_set is None:
        print('epoch=', epoch)
        root_dir = '/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
        if model_type == 'pretrain':
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
        elif model_type == 'finetune':
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + str(epoch) + '.model'))
        else:
            print('error: Model does not exist')

    print('end load model')

    return model


def calculate_bias_by_epoch(target_words, topic_words, media_name, random_seed_pretrain, random_seed_finetune=None, epoch_k=None, year_set=None):
    """
    :param target_words:
    :param topic_words:
    :param media_name:
    :param random_seed_pretrain:
    :param random_seed_finetune:
    :param epoch_k:
    :return:
    """
    # (1) 加载词向量模型
    if epoch_k == 0:  # 微调0个epoch == 预训练模型
        model_w2v = load_embed_model_by_mediaName('base', 'pretrain', random_seed_pretrain).wv
    elif epoch_k > 0:
        model_w2v = load_embed_model_by_mediaName(media_name, 'finetune', random_seed_pretrain, random_seed_finetune, epoch_k, seg_id=None, year_set=year_set).wv

    # (2) 加载target words和topic words对应的primary embedding
    target_word_primary_embedding = [model_w2v[word] for word in target_words]
    topic_word_primary_embedding = [model_w2v[word] for word in topic_words]

    # (3) 计算每个target word到topic words group的余弦距离
    bias_matrix = cosine_similarity(target_word_primary_embedding, topic_word_primary_embedding)  # [n1, n2], 每行对应1个target word到n2个topic words的cos_sim distance

    # (3) 统计最终的media bias值
    #     media bias = sum([:n1, :n2/2]) - sum([:n1, n2/2:])
    polar_size = int(len(topic_words) / 2)
    if polar_size * 2 != len(topic_words):  # 要求两个topic words group中包含的单词数量必须相等 (这种做法可能不妥当) 【X】
        print('error: capacity of topic words group1 is not equal to that of group2')
        return 'error'
    bias_sum_g1 = bias_matrix[:, :polar_size].sum(axis=1)  # axis=0表示列, axis=1表示行
    bias_sum_g2 = bias_matrix[:, polar_size:].sum(axis=1)
    final_bias = bias_sum_g1 - bias_sum_g2  # sum of Sim_1  -  sum of Sim_2
    final_bias /= polar_size  # 平均

    del model_w2v  # 释放资源
    return bias_matrix, bias_sum_g1, bias_sum_g2, final_bias


def calculate_bias_all_epoch(target_words, topic_words, media_name, seeds=None, epoch_k_list=None, year_set=None):
    """
    针对一个媒体，计算各个微调epoch后的bias，用于绘制曲线
    :param target_words:
    :param topic_words:
    :param media_name:
    :param random_seed_pretrain:
    :param random_seed_finetune:
    :param epoch_k_list:
    :return:
    """
    # 1. 统计各个微调epoch后的bias
    target_2_epoch_bias = defaultdict(dict)
    for r_seed in seeds:
        for epoch_k in epoch_k_list:
            bias_matrix, bias_sum_g1, bias_sum_g2, final_bias = calculate_bias_by_epoch(target_words, topic_words, media_name, r_seed, r_seed, epoch_k, year_set)
            for idx in range(0, len(target_words)):
                target_word = target_words[idx]
                cur_bias = final_bias[idx]
                target_2_epoch_bias[target_word][r_seed] = cur_bias

    # 2. 绘制 target:epoch_k~bias曲线 , 每个target word一个子图
    pos_label = topic_words[0]
    neg_label = topic_words[int(len(topic_words)/2)]
    # util_draw.draw_bias_curve_subplots_new(media_name, target_2_epoch_bias, pos_label, neg_label)

    # 3. 平均最后12次微调的bias，作为一个target word的最终bias
    target_bias_list = []  # len=len(target_words)
    for target_word in target_words:
        epoch_2_bias = target_2_epoch_bias[target_word]
        bias_avg = np.array(list(epoch_2_bias.values())[:]).mean()  # 对最后10次微调的结果求平均值
        target_bias_list.append(bias_avg)

    print('calculate down')
    return target_2_epoch_bias, target_bias_list


if __name__ == '__main__':
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']

    target_words_t1 = ['police', 'driver', 'lawyer', 'director', 'scientist', 'photographer', 'teacher', 'nurse']
    topic_words_t1 = ['man', 'male', 'brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him', 'woman', 'female', 'sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']

    target_words_t2 = ['virus', 'pandemic']
    topic_words_t2 = ['china', 'chinese', 'wuhan', 'beijing', 'shanghai', 'america', 'american', 'newyork', 'losangeles', 'chicago']
    # topic_words_t2 = ['china', 'chinese', 'wuhan', 'beijing', 'shanghai', 'guangzhou', 'shenzhen', 'america', 'american', 'newyork', 'washington', 'losangeles', 'boston', 'chicago']

    target_words_t3 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t3 = ['rich', 'wealthy', 'affluent', 'prosperous', 'plentiful', 'poor', 'impoverished', 'needy', 'penniless', 'miserable']

    target_words_t4 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t4 = ['education', 'learned', 'educated', 'professional', 'elite', 'ignorance', 'foolish', 'rude', 'folly', 'ignorant']

    states, states_top10 = util_corpus.read_usa_states()
    target_words_t5 = states_top10
    topic_words_t5 = ['republican', 'conservative', 'tradition', 'republic', 'gop', 'democrat', 'radical', 'revolution', 'liberal', 'democratic']

    # 1. 绘制bias曲线
    N = 30  # 12*5
    target_words = target_words_t2
    topic_words = topic_words_t2
    # random_seed_pretrain = 28
    # random_seed_finetune = 28
    seeds = list(range(28, 33))
    epoch_k_list = [N]

    year_sets = [[2016, 2017, 2018, 2019], [2020, 2021]]
    # media_target_matrix_years = []
    # for year_set in year_sets:
    #
    #     media_2_bias_data = {}
    #     media_target_matrix = []
    #     test_media_subset = media_list[:12]
    #     # 绘制bias曲线(1)
    #     for media_name in test_media_subset:
    #         target_2_epoch_bias, target_avg_bias_list = calculate_bias_all_epoch(target_words, topic_words, media_name, random_seed_pretrain, random_seed_finetune, epoch_k_list, year_set)
    #         media_2_bias_data[media_name] = target_2_epoch_bias
    #         media_target_matrix.extend(target_avg_bias_list)
    #     # 绘制bias heatmap
    #     s1 = len(test_media_subset)  # Debug: 变更 test_media_subset , 观察结果
    #     s2 = len(target_words)
    #     pos_label = topic_words[0]
    #     neg_label = topic_words[int(len(topic_words)/2)]
    #     media_target_matrix = np.array(media_target_matrix[:(s1*s2)]).reshape(s1, s2)
    #     figsize = (max(1, int(len(test_media_subset) / 30)) * 13, max(1, int(len(target_words) / 30)) * 10)
    #     colorbar_boundary = math.ceil(max(abs(media_target_matrix.min()), abs(media_target_matrix.max())) / 0.01 + 0) * 0.01  # heatmap-colorbar的取值区间    winter plasma
    #     util_draw.draw_pairwise_matrix_2(np.around(media_target_matrix, 3).T, target_words, test_media_subset, pos_label, neg_label, figsize, 13, threshold=0, cmap='winter', vmin=(-1)*colorbar_boundary, vmax=colorbar_boundary, year_set=year_set)
    #     media_target_matrix_years.append(media_target_matrix)  # 保存数据
    # 绘制差值
    # util_draw.draw_pairwise_matrix_2(np.around(media_target_matrix_years[1], 3).T - np.around(media_target_matrix_years[0], 3).T, target_words, test_media_subset, pos_label, neg_label, figsize, 13, threshold=0, cmap='winter', vmin=(-1) * colorbar_boundary, vmax=colorbar_boundary, year_set=year_set)

    # (2) 将 covid-country 和 occupation-gender 放在一起
    media_target_matrix_list = []
    target_words = target_words_t2 + target_words_t1
    for year_set in year_sets:
        media_2_bias_data = {}
        media_target_matrix = []
        test_media_subset = media_list[:]
        # 绘制bias曲线(1)
        for media_name in test_media_subset:
            target_2_epoch_bias, target_avg_bias_list_2 = calculate_bias_all_epoch(target_words_t2, topic_words_t2, media_name, seeds, epoch_k_list, year_set)
            target_2_epoch_bias, target_avg_bias_list_1 = calculate_bias_all_epoch(target_words_t1, topic_words_t1, media_name, seeds, epoch_k_list, year_set)
            media_target_matrix.extend(list(target_avg_bias_list_2) + list(target_avg_bias_list_1))
        # 绘制bias heatmap
        s1 = len(test_media_subset)  # Debug: 变更 test_media_subset , 观察结果
        s2 = len(target_words)
        pos_label = topic_words[0]
        neg_label = topic_words[int(len(topic_words) / 2)]
        media_target_matrix = np.array(media_target_matrix[:(s1 * s2)]).reshape(s1, s2)
        figsize = (max(1, int(len(test_media_subset) / 30)) * 13, max(1, int(len(target_words) / 30)) * 9)
        colorbar_boundary = math.ceil(max(abs(media_target_matrix.min()), abs(media_target_matrix.max())) / 0.01 + 0) * 0.01  # heatmap-colorbar的取值区间    winter plasma
        media_target_matrix_list.append(media_target_matrix)
    pos_label = 'china\\man'
    neg_label = 'america\\woman'
    thr = -1
    media_target_matrix_delta = np.around(media_target_matrix_list[1], 3).T - np.around(media_target_matrix_list[0], 3).T
    for i in range(0, media_target_matrix_delta.shape[0]):
        for j in range(0, media_target_matrix_delta.shape[1]):
            bias_item = media_target_matrix_delta[i][j]
            if abs(bias_item) < thr:
                media_target_matrix_delta[i][j] = 0
            else:
                continue
    util_draw.draw_pairwise_matrix_2(media_target_matrix_delta, target_words, test_media_subset, neg_label, pos_label, figsize, 12, threshold=0, cmap='winter', vmin=(-1) * 0.12, vmax=0.12)

    print('test down')

