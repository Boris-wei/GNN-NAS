import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os

def proxy_vec_analysis():
    methods = ['golden', 'bad']
    poolings = ['add', 'mean', 'max']
    num_vector = 10
    for method in methods:
        for pooling in poolings:
            for i in range(0, num_vector):
                for j in range(i + 1, num_vector):
                    name1 = str(i)
                    file_name1 = 'proxy_all_' + pooling + '_' + method + '_' + name1 + '.pt'
                    path_name1 = os.path.join('datasets/ogbg_molhiv/', file_name1)
                    proxy_vec1 = torch.load(path_name1)
                    name2 = str(j)
                    file_name2 = 'proxy_all_' + pooling + '_' + method + '_' + name2 + '.pt'
                    path_name2 = os.path.join('datasets/ogbg_molhiv/', file_name2)
                    proxy_vec2 = torch.load(path_name2)
                    similarity = F.cosine_similarity(proxy_vec1, proxy_vec2, dim=1)
                    similarity = similarity.cpu()
                    similarity = similarity.detach().numpy()
                    plt.hist(similarity, bins=30, histtype='bar', color='orange', rwidth=0.9)   # 直方统计函数，bins表示区间数量，histtype表示直方图形式
                    plt.xlim(0, 1)
                    plt.ylim(0, 10000)
                    plt.xlabel('cosine_similarity')
                    plt.ylabel('number of graphs in the dataset')
                    title = pooling + '_' + method + '_' + name1 + '_' + name2
                    plt.title(title)
                    fig_name = './figures/' + title + '.png'
                    plt.savefig(fig_name)
                    plt.show()


def random_vec_analysis(method='cos'):
    num_vector = 10
    for i in range(0, num_vector):
        for j in range(i+1, num_vector):
            name1 = str(i)
            file_name1 = 'rand_all_' + name1 + '.pt'
            path_name1 = os.path.join('datasets/ogbg_molhiv/', file_name1)
            rand_vec1 = torch.load(path_name1)
            name2 = str(j)
            file_name2 = 'rand_all_' + name2 + '.pt'
            path_name2 = os.path.join('datasets/ogbg_molhiv/', file_name2)
            rand_vec2 = torch.load(path_name2)
            similarity = F.cosine_similarity(rand_vec1, rand_vec2, dim=1)
            if method == 'angle':
                angle = torch.acos(similarity)
                angle = angle.cpu()
                angle = angle.detach().numpy()
                plt.hist(angle, bins=30, histtype='bar', color='orange',
                         rwidth=0.9)  # 直方统计函数，bins表示区间数量，histtype表示直方图形式
                plt.xlim(0, torch.pi / 2)
                plt.ylim(0, 10000)
                plt.xlabel('cosine_similarity (angle)')
                plt.ylabel('number of graphs in the dataset')
                title = 'rand_all_angle_' + name1 + '_' + name2
                plt.title(title)
                fig_name = './figures/' + title + '.pdf'
                plt.savefig(fig_name)
                plt.show()
            elif method == 'cos_value':
                similarity = similarity.cpu()
                similarity = similarity.detach().numpy()
                plt.hist(similarity, bins=30, histtype='bar', color='orange',
                         rwidth=0.9)  # 直方统计函数，bins表示区间数量，histtype表示直方图形式
                plt.xlim(0, 1)
                plt.ylim(0, 10000)
                plt.xlabel('cosine_similarity')
                plt.ylabel('number of graphs in the dataset')
                title = 'rand_all_' + name1 + '_' + name2
                plt.title(title)
                fig_name = './figures/' + title + '.png'
                plt.savefig(fig_name)
                plt.show()


if __name__ == '__main__':
    analyze_method = 'random'
    if analyze_method == 'proxy':
        proxy_vec_analysis()
    elif analyze_method == 'random':
        random_vec_analysis(method='angle')

