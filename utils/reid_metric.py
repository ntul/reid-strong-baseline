# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import shutil
import os
import math
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

class R1_mAP(Metric):
    def __init__(self, val_set, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.val_set = val_set

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = qf#feats[self.num_query:]
        g_pids = q_pids#np.asarray(self.pids[self.num_query:])
        g_camids = q_camids#np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        #for i,x in enumerate(self.val_loader):
        #    if(i>2):
        #        break
        #    print(x)

        if(self.val_set!=None):
            def mkdir_if_missing(dirname):
                if not os.path.exists(dirname):
                    try:
                        os.makedirs(dirname)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise

            model = AgglomerativeClustering(affinity='precomputed', n_clusters=750, linkage='complete').fit(distmat)
            labels = model.labels_
            indices = []

            for i,x in enumerate(labels):
                indices.append([])
                for i2,x2 in enumerate(labels):
                    if(x!=-1 and x==x2 and i2!=0): 
                        indices[i].append(i2)
                        labels[i2]=-1

            indices[0].append(0)
            indices = list(filter(None, indices))


            dst = './clusters'

            if os.path.exists(dst) and os.path.isdir(dst):
                shutil.rmtree(dst)


            for n,i in enumerate(indices):
                dstt = os.path.join(dst, str(n)+'x') #dataset[0][i[0]][0].split('/')[-1][0:4])
                mkdir_if_missing(dstt)
                for j in i:
                    shutil.copy(self.val_set[j][3], dstt)

            for i in os.listdir(dst):
                listt = list(map(lambda x:x[0:4], os.listdir(os.path.join(dst,i))))
                dict = {j : listt.count(j) for j in listt}
                name = sorted(dict.items(), key = lambda x:x[1])[-1][0]
                if(os.path.exists(os.path.join(dst,name+'_4'))):
                    os.rename(os.path.join(dst,i),os.path.join(dst,name+'_5'))
                elif(os.path.exists(os.path.join(dst,name+'_3'))):
                    os.rename(os.path.join(dst,i),os.path.join(dst,name+'_4'))
                elif(os.path.exists(os.path.join(dst,name+'_2'))):
                    os.rename(os.path.join(dst,i),os.path.join(dst,name+'_3'))
                elif(os.path.exists(os.path.join(dst,name))):
                    os.rename(os.path.join(dst,i),os.path.join(dst,name+'_2'))      
                else:
                    os.rename(os.path.join(dst,i),os.path.join(dst,name))

            print("Calculating custom metric...")
            TP = 0
            FP = 0
            FN = 0
            for i in os.listdir(dst):
                listt = list(map(lambda x:x[0:4], os.listdir(os.path.join(dst,i))))
                correct = listt.count(i[0:4])
                TP += correct
                FP += len(listt)-correct
                queryList = list(map(lambda x:x[0:4], os.listdir('./market1501/query')))
                FN += queryList.count(i)-correct

            FMI = TP / math.sqrt((TP + FP) * (TP + FN))
            print("TP: "+str(TP))
            print("FP: "+str(FP))
            print("FN: "+str(FN))
            print("Fowlkes-Mallows Score: "+str(FMI))
        
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP