import json
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
# from syntaticeval import weightedrouge


coherence=[]
consistency=[]
fluency=[]
relevance=[]

with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		if obj['model_id']=='M8':
			
			decoded=obj['decoded']
			t1=obj['expert_annotations'][0]
			t2=obj['expert_annotations'][1]
			t3=obj['expert_annotations'][2]
			# consistency': 1, 'fluency': 4, 'relevance': 2
			coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
			consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
			fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
			relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
			coherence.append(coherenceval)
			consistency.append(consistencyval)
			fluency.append(fluencyval)
			relevance.append(relevanceval)


# print(coherence)
# print(len(coherence))



corr1, _ = kendalltau(coherence,coherence)
corr2, _ = kendalltau(coherence,consistency)
corr3, _ = kendalltau(coherence,fluency)
corr4, _ = kendalltau(coherence,relevance)

corr5, _ = kendalltau(consistency,coherence)
corr6, _ = kendalltau(consistency,consistency)
corr7, _ = kendalltau(consistency,fluency)
corr8, _ = kendalltau(consistency,relevance)

corr9, _ = kendalltau(fluency,coherence)
corr10, _ = kendalltau(fluency,consistency)
corr11, _ = kendalltau(fluency,fluency)
corr12, _ = kendalltau(fluency,relevance)

corr13, _ = kendalltau(relevance,coherence)
corr14, _ = kendalltau(relevance,consistency)
corr15, _ = kendalltau(relevance,fluency)
corr16, _ = kendalltau(relevance,relevance)

print('------------- | coherence | consistency | fluency| relevance')
print('coherence     |  {:.2f}   |    {:.2f}   |     {:.2f}   |     {:.2f}   |'.format(corr1,corr2,corr3,corr4))
print('consistency   |  {:.2f}   |    {:.2f}   |     {:.2f}   |     {:.2f}   |'.format(corr5,corr6,corr7,corr8))
print('fluency       |  {:.2f}   |    {:.2f}   |     {:.2f}   |     {:.2f}   |'.format(corr9,corr10,corr11,corr12))
print('relevance     |  {:.2f}   |    {:.2f}   |     {:.2f}   |     {:.2f}   |'.format(corr13,corr14,corr15,corr16))