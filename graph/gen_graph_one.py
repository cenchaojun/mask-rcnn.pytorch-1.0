#from torch.utils.data import Dataset, DataLoader
#import cv2
#from torchvision import models, transforms
import pdb
import json
import os
#import torch
import sys
import argparse
import numpy as np  
# import networkx as nx
from xlwt import * 
import math
import pdb

# 肺结核 肺实变 允许同时出现
# classes = ['_background_','肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',
#            '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张',
#            '多发结节', '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常',
#            '肺结核', '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常']
#
# graph_classes = ['多发结节','弥漫性结节','空洞','纤维化表现','结节','肺不张','肺实变','肺结核','肿块','胸腔积液']

# cvpr code
# classes = ['_background_', '肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '结节',
#               '肿块', '气胸', '肺气肿', '钙化', '弥漫性结节', '肺不张',
#               '心影增大', '骨折']  # 13
#
# graph_classes = ['弥漫性结节', '纤维化表现', '结节', '肺不张', '肺实变', '肿块', '胸腔积液']

# 竞赛
classes = ['_background_', '肺实变', '纤维化表现', '胸腔积液', '结节',
                      '肿块', '气胸', '肺气肿', '钙化', '肺不张', '骨折']

graph_classes = ['纤维化表现', '结节', '肺不张', '肺实变', '肿块', '胸腔积液']

def gen_graph():
    # json_file = '/data1/DX/train_anno_DX.json'
    # json_file = '/data1/DX/total_annos.json'
    # json_file = '/data1/liujingyu/DR/total_annos.json'
    test_json_file = '/home/lianjie/cvpr_code/part_seg/yy_jsons/test_gk_yy.json'
    train_json_file = '/home/lianjie/cvpr_code/part_seg/yy_jsons/train_gk_yy.json'

    assert os.path.exists(test_json_file), 'Json path does not exist: {}'.format(test_json_file)
    test_anno = json.load(open(test_json_file))
    train_anno = json.load(open(train_json_file))
    train_anno.extend(test_anno)
    
    class_num = len(classes)
    graph = np.zeros((class_num,class_num),dtype=np.int64)

    book = Workbook(encoding='utf-8')
    sheet = book.add_sheet('graph')
    sheet_norm = book.add_sheet('graph_norm')
    for i in range(1,len(classes)+1):
        sheet.write(0,i,label = classes[i-1])
        sheet.write(i,0,label = classes[i-1])
        sheet_norm.write(0,i,label = classes[i-1])
        sheet_norm.write(i,0,label = classes[i-1])


    for entry in train_anno:
        result = []
        syms = entry['syms']
        for _, sym in enumerate(syms):
            if '膈面异常' in sym and entry['doc_name'] == 'fj6311':
                continue

            if '主动脉异常' in sym and '钙化' in sym:
                sym = ['主动脉钙化', '主动脉异常']
            if '结节' in sym and '乳头影' in sym:  # 费主任标了好多这种，结节和乳头影都在，我们认为是乳头影
                sym = ['乳头影']

            if '结节' in sym and '弥漫性结节' in sym:
                sym.remove('结节')
            if '结节' in sym and '多发结节' in sym:
                sym.remove('结节')
            if '结核结节' in sym and '弥漫性结节' in sym:
                sym.remove('结核结节')
            if '结核结节' in sym and '多发结节' in sym:
                sym.remove('结核结节')
            if '结核球' in sym and '弥漫性结节' in sym:
                sym.remove('结核球')
            if '结核球' in sym and '多发结节' in sym:
                sym.remove('结核球')

            for s in sym:  # for each sub-sym
                if s == '膈面膨隆' or s == '膈面抬高':  # awkward ...
                    s = '膈面异常'
 
                if s == '肺门影浓' or s == '肺门影大':
                    s = '肺门增浓'

                if s == '主动脉异常':
                    s = '主动脉结增宽'

                # 以下是肺结核的征象
                if s == '三均匀粟粒样结节' or s == '非三均匀粟粒样结节':
                    s = '弥漫性结节'

                if s == '结核球' or s == '结核结节':
                    s = '结节'

                if s == '索条影':
                    s = '纤维化表现'

                # cvpr code
                if s == '骨折' or s == '肋骨缺失':
                    s = '骨折'
                if s == '弥漫性结节' or s == '多发结节':
                    s = '弥漫性结节'

                result.append(s)

        # pdb.set_trace()
        # solution = flatten()
        # syms = entry['syms']
        # result = solution.get_list(syms)
        result = list(set(result)) #去除重复元素
        c_index = []
        for i in range(len(result)):
            if result[i] not in classes: continue
            c_index.append(classes.index(result[i]))
        c_index.sort()
        
        if len(c_index) == 0:
            continue
        elif len(c_index) == 1: #主对角元素统计
            raw = c_index[0]
            column = c_index[0]
            graph[raw][column] += 1
        
        else:
            for i in range(len(c_index)): #主对角元素统计
                raw = c_index[i]
                column = c_index[i]
                graph[raw][column] += 1
            
            for j in range(len(c_index)-1): #对角线上方元素统计 最后做转置相加
                k = j+1
                while k<(len(c_index)):
                    raw = c_index[j]
                    column = c_index[k]
                    # 加上判断语句 只对特定几类 建立图关系
                    if classes[raw] in graph_classes and classes[column] in graph_classes:
                        graph[raw][column] += 1
                    k += 1

    graph = graph + graph.transpose()
    for i in range(len(graph)):
        graph[i][i] = graph[i][i]//2
    # frequent     
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            sheet.write(i+1,j+1,label = int(graph[i][j]))

    graph[0][0] = 1000 # background random value (norm -> 1.0)

    # normalize graph
    graph_norm = np.zeros((len(classes),len(classes)),dtype=np.float64)
    for i in range(len(graph)): #ingore background
        for j in range(len(graph[0])):
            if graph[i][i] == 0 or graph[j][j] ==0: # 不存在的类
                graph_norm[i][j] = 0.0
                pdb.set_trace()
            else:
                graph_norm[i][j] = graph[i][j]/math.sqrt(graph[i][i]*graph[j][j])

            sheet_norm.write(i+1,j+1,label = float(round(graph_norm[i][j],2)))

    # book.save('graph_all_yy.xls')
    np.save('graph_gk_yy10.npy',graph_norm)

class flatten(object):
    def __init__(self):
        self.flatten_list = []

    def get_list(self,syms):
        for item in syms:
            if isinstance(item,list):
                self.get_list(item)
            else:
                self.flatten_list.append(item)

        return self.flatten_list

if __name__ == '__main__':
    gen_graph()
