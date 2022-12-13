#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 2_1_word2vec_bias_epoch_coarse_grain.py
# @Author: Hua Zhu
# @Date  : 2022/4/16
# @Desc  : 监控分析, 每次微调之后，bias的变化
import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import util_draw
import pickle
import util_corpus


def load_embed_model_by_mediaName(media_name, model_type, random_seed_pretrain=None, random_seed_finetuned=None, epoch_k=None):
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
    root_dir = '/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
    if model_type == 'pretrain':
        model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
    elif model_type == 'finetune':
        model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + str(epoch_k) + '.model'))
    else:
        print('error: Model does not exist')
    print('end load model')

    return model


def calculate_bias_by_epoch(target_words, topic_words, media_name, random_seed_pretrain, random_seed_finetune=None, epoch_k=None):
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
        model_w2v = load_embed_model_by_mediaName(media_name, 'finetune', random_seed_pretrain, random_seed_finetune, epoch_k).wv

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


def calculate_bias_all_epoch(target_words, topic_words, media_name, seeds=None, epoch_k_list=None):
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
    target_2_seed_bias = defaultdict(dict)

    for r_seed in seeds:
        for epoch_k in epoch_k_list:
            print('epoch=', epoch_k)
            bias_matrix, bias_sum_g1, bias_sum_g2, final_bias = calculate_bias_by_epoch(target_words, topic_words, media_name, r_seed, r_seed, epoch_k)
            for idx in range(0, len(target_words)):
                target_word = target_words[idx]
                cur_bias = final_bias[idx]
                target_2_seed_bias[target_word][r_seed] = cur_bias

    # 2. 绘制 target:epoch_k~bias曲线 , 每个target word一个子图
    pos_label = topic_words[0]
    neg_label = topic_words[int(len(topic_words)/2)]
    # util_draw.draw_bias_curve_subplots_new(media_name, target_2_epoch_bias, pos_label, neg_label)

    # 3. 平均最后12次微调的bias，作为一个target word的最终bias
    target_bias_list = []  # len=len(target_words)
    for target_word in target_words:
        seed_2_bias = target_2_seed_bias[target_word]
        bias_avg = np.array(list(seed_2_bias.values())[:]).mean()  # 对最后12次微调的结果求平均值
        target_bias_list.append(bias_avg)

    print('calculate down')
    return target_2_seed_bias, target_bias_list


def save_media_target_matrix(random_seed, topic_words, media_target_matrix):
    """
    保存 media_target_matrix ，用于统计学分析
    :param random_seed:
    :param topic_words:
    :param media_target_matrix:
    :return:
    """
    save_dir = '/home/newsgrid/zhuhua/NHB/data/bias_random/'
    file_name = str(random_seed) + '_' + topic_words[0] + '_' + topic_words[int(len(topic_words)/2)]
    with open(save_dir+file_name, 'wb') as save_f:
        pickle.dump(media_target_matrix, save_f)

    print('save media_target_matrix down')


def load_media_target_matrix(random_seed, topic_words):
    save_dir = '/home/newsgrid/zhuhua/NHB/data/bias_random/'
    file_name = str(random_seed) + '_' + topic_words[0] + '_' + topic_words[int(len(topic_words)/2)]
    with open(save_dir+file_name, 'rb') as read_f:
        res_obj = pickle.load(read_f)

    print('load media_target_matrix down')
    return res_obj


if __name__ == '__main__':
    target_words_t1 = ['police', 'driver', 'lawyer', 'director', 'scientist', 'photographer', 'teacher', 'nurse']
    topic_words_t1 = ['man', 'male', 'brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him', 'woman', 'female', 'sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']

    target_words_t2 = ['covid', 'coronavirus', 'virus', 'pandemic', 'omicron']
    topic_words_t2 = ['china', 'chinese', 'wuhan', 'beijing', 'shanghai', 'america', 'american', 'newyork', 'losangeles', 'chicago']

    target_words_t3 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t3 = ['rich', 'wealthy', 'affluent', 'prosperous', 'plentiful', 'poor', 'impoverished', 'needy', 'penniless', 'miserable']

    states, states_top10 = util_corpus.read_usa_states()
    target_words_t4 = states_top10
    topic_words_t4 = ['republican', 'conservative', 'tradition', 'republic', 'gop', 'democrat', 'radical', 'revolution', 'liberal', 'democratic']

    # target_words_t5 = ['asian', 'african', 'hispanic', 'latino']
    # topic_words_t5 = ['education', 'learned', 'educated', 'professional', 'elite', 'ignorance', 'foolish', 'rude', 'folly', 'ignorant']

    # 1. 绘制bias曲线
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    N = 30  # 12*5 / 2

    t_indexs = [1, 2, 3, 4]
    seeds = list(range(28, 33))  # fig5
    # seeds = list(range(33, 38))
    target_words = target_words_t4
    topic_words = topic_words_t4
    # random_seed_pretrain = 32
    # random_seed_finetune = 32
    # epoch_k_list = list(range(0, N+1))  # 考虑fine-tune最后的几个epoch
    epoch_k_list = [N]  # 只考虑fine-tune的最后一个epoch (for t-test)

    media_2_bias_data = {}
    media_target_matrix = []
    test_media_subset = media_list[:12]
    # 绘制bias曲线(1)
    for media_name in test_media_subset:
        target_2_epoch_bias, target_avg_bias_list = calculate_bias_all_epoch(target_words, topic_words, media_name, seeds, epoch_k_list)
        media_2_bias_data[media_name] = target_2_epoch_bias
        media_target_matrix.extend(target_avg_bias_list)
    # 绘制bias heatmaps
    s1 = len(test_media_subset)  # Debug: 变更 test_media_subset , 观察结果
    s2 = len(target_words)
    pos_label = topic_words[0]
    neg_label = topic_words[int(len(topic_words)/2)]
    media_target_matrix = np.array(media_target_matrix[:(s1*s2)]).reshape(s1, s2)
    figsize = (max(1, int(len(test_media_subset) / 30)) * 13, max(1, int(len(target_words) / 30)) * 10)
    colorbar_boundary = math.ceil(max(abs(media_target_matrix.min()), abs(media_target_matrix.max())) / 0.01 + 0) * 0.01  # heatmap-colorbar的取值区间    winter plasma
    util_draw.draw_pairwise_matrix_2(np.around(media_target_matrix, 3).T, target_words, test_media_subset, pos_label, neg_label, figsize, 9, threshold=0, cmap='winter', vmin=(-1)*colorbar_boundary, vmax=colorbar_boundary)
    # 绘制bias曲线(2)
    # colors = ['#ff0000', '#ff6100', '#ffff00', '#00c957', '#082e54', '#87ceeb', '#385e0f', '#8a2be2', '#ff7d40', '#f0e68c', '#bc8f8f', '#c76114']
    # styles = ['.-']*12
    # util_draw.draw_bias_curve_subplots_multiple_media(media_list[:12], media_2_bias_data, target_words, pos_label, neg_label, styles, colors)
    # util_draw.draw_bias_curve_subplots_multiple_media(media_list[:6], media_2_bias_data, target_words, pos_label, neg_label, styles, colors)
    # util_draw.draw_bias_curve_subplots_multiple_media(media_list[6:], media_2_bias_data, target_words, pos_label, neg_label, styles, colors)

    # 2. 保存 media_target_matrix ，用于统计学分析
    # save_media_target_matrix(random_seed_finetune, topic_words, media_target_matrix)
    # print('media_target_matrix.shape = ', media_target_matrix.shape)

    print('test down')
