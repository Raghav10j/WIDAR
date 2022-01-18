
import json
import time
import jsonlines
import pickle
import six
# from eval import compute_rouge_n,tokenizef,_summary_level_lcs,sentencelevelrougeL

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from rouge_score import rouge_scorer
import statistics
from scipy.stats import kendalltau as correlator
from syntaticeval import *
from statistics import mean


# # OGrougerecalllist=[]
# # OGrougeprecisionlist=[]
# OGrougefscorelist=[]
# OGcoherencelist=[]
# OGconsistencylist=[]
# OGfluencylist=[]
# OGrelevancelist=[]
# iq=0
# with jsonlines.open('model_annotations.aligned.scored.jsonl') as reader:
# 	for obj in reader:
# 		if obj['model_id']!='M23_dynamicmix':
# 			print(iq)
# 			iq=iq+1
# 			# OGrougerecalllist.append(obj['metric_scores_11']['bert_score_f1'])
# 			# OGrougeprecisionlist.append(obj['metric_scores_11']['bert_score_f1'])
# 			OGrougefscorelist.append(obj['metric_scores_11']['rouge_we_1_p'])
# 			# OGrougerecalllist.append(obj['metric_scores_11']['bert_score_recall'])
# 			# OGrougeprecisionlist.append(obj['metric_scores_11']['bert_score_precision'])
# 			# OGrougefscorelist.append(obj['metric_scores_11']['bert_score_f1'])
# 			t1=obj['expert_annotations'][0]
# 			t2=obj['expert_annotations'][1]
# 			t3=obj['expert_annotations'][2]
# 			# consistency': 1, 'fluency': 4, 'relevance': 2
# 			coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
# 			consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
# 			fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
# 			relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
# 			OGcoherencelist.append(coherenceval)
# 			OGconsistencylist.append(consistencyval)
# 			OGfluencylist.append(fluencyval)
# 			OGrelevancelist.append(relevanceval)



# print('----------------fscore---------------')
# corr2,_=kendalltau(OGcoherencelist,OGrougefscorelist)
# print('Kendall Rank correlation b/w coherence and rouge_l_f: {:.3f}'.format(corr2))

# corr4,_=kendalltau(OGconsistencylist,OGrougefscorelist)
# print('kendall Rank correlation b/w consistency and bleu" fscore: {:.3f}'.format(corr4))

# corr6,_=kendalltau(OGfluencylist,OGrougefscorelist)
# print('kenda rank correlation b/w fluency and blanc fscore: {:.3f}'.format(corr6))

# corr8,_=kendalltau(OGrelevancelist,OGrougefscorelist)
# print('kenda rank correlation b/w relevance and blanc fscore: {:.3f}'.format(corr8))


import pickle

with open("lamda/relevancefinalevalF1L.txt", "rb") as fp:
	b = pickle.load(fp)

# print(b)




OGrougerecalllist=[]
OGrougeprecisionlist=[]
OGrougefscorelist=[]
OGcoherencelist=[]
OGconsistencylist=[]
OGfluencylist=[]
OGrelevancelist=[]
iq=0
with jsonlines.open('model_annotations.aligned.scored.jsonl') as reader:
	for obj in reader:
		if obj['model_id']!='M23_dynamicmix':
			print(iq)
			iq=iq+1
			OGrougerecalllist.append(obj['metric_scores_11']['rouge']['rouge_l_recall'])
			OGrougeprecisionlist.append(obj['metric_scores_11']['rouge']['rouge_l_precision'])
			OGrougefscorelist.append(obj['metric_scores_11']['rouge']['rouge_l_f_score'])
			# OGrougerecalllist.append(obj['metric_scores_11']['bert_score_recall'])
			# OGrougeprecisionlist.append(obj['metric_scores_11']['bert_score_precision'])
			# OGrougefscorelist.append(obj['metric_scores_11']['bert_score_f1'])
			t1=obj['expert_annotations'][0]
			t2=obj['expert_annotations'][1]
			t3=obj['expert_annotations'][2]
			# consistency': 1, 'fluency': 4, 'relevance': 2
			coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
			consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
			fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
			relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
			OGcoherencelist.append(coherenceval)
			OGconsistencylist.append(consistencyval)
			OGfluencylist.append(fluencyval)
			OGrelevancelist.append(relevanceval)

corr,_=kendalltau(OGrelevancelist,OGrougefscorelist)
print(corr)



import matplotlib.pyplot as plt
import numpy as np
# plt.xlabel('value of λ')
# plt.ylabel('correlation with consistency')
y=np.array(b)
x=np.arange(0, 1.1, 0.1)
print(y)

plt.plot(x, y)  # Plot the chart
x2=x
y2=np.array([corr for i in range(len(b))])
print(y2)
plt.plot(x2,y2,'--')

# plt.legend([])
x3=x
y3=np.array([0.234 for i in range(len(b))])
plt.plot(x3,y3,'-.')
x4=x
y4=np.array([0.222 for i in range(len(b))])
plt.plot(x4,y4,':')
plt.legend(['Fixed λ','ROUGE','λ=max(Wcov)','λ=mean(Wcov)'],prop={'size': 12})
plt.title("Relevance",fontdict={'fontsize': 15})
plt.show()  # display
plt.savefig('lambda2/relevance.png')
plt.savefig('lambda2/relevance.pdf')