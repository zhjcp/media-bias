#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_draw.py
# @Author: Hua Zhu
# @Date  : 2022/3/31
# @Desc  : 绘图工具类

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def draw_matrix(time_period, country_num, data_matrix, x_labels, y_labels, x_size=13, y_size=10, label_fz=10, text_fz=10):
    figsize = x_size, y_size
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_matrix)

    plt.xticks(fontsize=label_fz)
    plt.yticks(fontsize=label_fz)
    plt.xticks(fontsize=label_fz)
    plt.yticks(fontsize=label_fz)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor", fontsize=label_fz)

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, data_matrix[i, j], ha="center", va="center", color="w", font='DejaVu Sans', fontsize=text_fz)

    ax.set_title(time_period + ': ' + "pairwise similarity between sources from different countries", font='DejaVu Sans')
    fig.tight_layout()

    plt.savefig('/home/newsgrid/zhuhua/NHB/experiment/selection_bias/results/pairwise_sim/' + time_period + ' _ ' + str(country_num) + '.png')
    plt.show()
