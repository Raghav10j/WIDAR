import json
import jsonlines
import pickle
import six
# from eval import compute_rouge_n,tokenizef,_summary_level_lcs,sentencelevelrougeL

from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import kendalltau

# from scipy.stats import 
from rouge_score import rouge_scorer
import statistics
from scipy.stats import kendalltau as correlator
from syntaticeval import *




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
			OGrougerecalllist.append(obj['metric_scores_11']['rouge']['rouge_2_recall'])
			OGrougeprecisionlist.append(obj['metric_scores_11']['rouge']['rouge_2_precision'])
			OGrougefscorelist.append(obj['metric_scores_11']['rouge']['rouge_2_f_score'])
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




rougerecalllist=[]
rougefscorelist=[]
rougePrecisionlist=[]
coherencelist=[]
consistencylist=[]
fluencylist=[]
relevancelist=[]


iq=0

with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		# text=obj['text']
		decoded=obj['decoded']
		print(iq)
		iq+=1
		t1=obj['expert_annotations'][0]
		t2=obj['expert_annotations'][1]
		t3=obj['expert_annotations'][2]
		# consistency': 1, 'fluency': 4, 'relevance': 2
		coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
		consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
		fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
		relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
		coherencelist.append(coherenceval)
		consistencylist.append(consistencyval)
		fluencylist.append(fluencyval)
		relevancelist.append(relevanceval)

		currrougerecall=0
		currrougeprecision=0
		currrougefscore=0
		for k in [0,1,2,3,4,5,6,7,8,9,10]:
			ref=obj['references'][k]
			article=obj['text']
			currrougetemprecall,currrougetempprecision,currrougetempfscore,x=weightedrouge(ref,decoded,article,2)
			# currrougetempfscore=currrougetempfscore/2
			# currrougetemprecall=currrougetemprecall/2
			# currrougetempprecision=currrougetempprecision/2


			currrougerecall=currrougerecall+currrougetemprecall
			currrougeprecision=currrougeprecision+currrougetempprecision
			currrougefscore=currrougefscore+currrougetempfscore
			# currrouge+=currrougetemp
		currrougerecall=currrougerecall/11
		rougerecalllist.append(currrougerecall)
		currrougeprecision=currrougeprecision/11
		rougePrecisionlist.append(currrougeprecision)
		currrougefscore=currrougefscore/11
		rougefscorelist.append(currrougefscore)
		# rougelist.append(currrouge)







# print('----------------without Wcov-----------------------')

print('----------------recall---------------')
corr1,_=kendalltau(coherencelist,rougerecalllist)
corr2,_=kendalltau(OGcoherencelist,OGrougerecalllist)
print('Kendall Rank correlation b/w coherence and our rouge-l recall: {:.3f}'.format(corr1))
print('Kendall Rank correlation b/w coherence and OG rouge-l recall: {:.3f}'.format(corr2))

corr3,_=kendalltau(consistencylist,rougerecalllist)
corr4,_=kendalltau(OGconsistencylist,OGrougerecalllist)
print('Kendall Rank correlation b/w consistency and our rouge-l recall: {:.3f}'.format(corr3))
print('kendall Rank correlation b/w consistency and OG rouge-l recall: {:.3f}'.format(corr4))

corr5,_=kendalltau(fluencylist,rougerecalllist)
corr6,_=kendalltau(OGfluencylist,OGrougerecalllist)
print('kenda rank correlation b/w fluency and our rouge-l recall: {:.3f}'.format(corr5))
print('kenda rank correlation b/w fluency and OG rouge-l recall: {:.3f}'.format(corr6))

corr7,_=kendalltau(relevancelist,rougerecalllist)
corr8,_=kendalltau(OGrelevancelist,OGrougerecalllist)
print('kenda rank correlation b/w relevance and our rouge-l recall: {:.3f}'.format(corr7))
print('kenda rank correlation b/w relevance and OG rouge-l recall: {:.3f}'.format(corr8))

print('----------------fscore---------------')
corr1,_=kendalltau(coherencelist,rougefscorelist)
corr2,_=kendalltau(OGcoherencelist,OGrougefscorelist)
print('Kendall Rank correlation b/w coherence and our rouge-l fscore: {:.3f}'.format(corr1))
print('Kendall Rank correlation b/w coherence and OG rouge-l fscore: {:.3f}'.format(corr2))

corr3,_=kendalltau(consistencylist,rougefscorelist)
corr4,_=kendalltau(OGconsistencylist,OGrougefscorelist)
print('Kendall Rank correlation b/w consistency and our rouge-l fscore: {:.3f}'.format(corr3))
print('kendall Rank correlation b/w consistency and OG rouge-l fscore: {:.3f}'.format(corr4))

corr5,_=kendalltau(fluencylist,rougefscorelist)
corr6,_=kendalltau(OGfluencylist,OGrougefscorelist)
print('kenda rank correlation b/w fluency and our rouge-l fscore: {:.3f}'.format(corr5))
print('kenda rank correlation b/w fluency and OG rouge-l fscore: {:.3f}'.format(corr6))

corr7,_=kendalltau(relevancelist,rougefscorelist)
corr8,_=kendalltau(OGrelevancelist,OGrougefscorelist)
print('kenda rank correlation b/w relevance and our rouge-l fscore: {:.3f}'.format(corr7))
print('kenda rank correlation b/w relevance and OG rouge-l fscore: {:.3f}'.format(corr8))



print('----------------precision---------------')
corr1,_=kendalltau(coherencelist,rougePrecisionlist)
corr2,_=kendalltau(OGcoherencelist,OGrougeprecisionlist)
print('Kendall Rank correlation b/w coherence and our rouge-l precision: {:.3f}'.format(corr1))
print('Kendall Rank correlation b/w coherence and OG rouge-l precision: {:.3f}'.format(corr2))

corr3,_=kendalltau(consistencylist,rougePrecisionlist)
corr4,_=kendalltau(OGconsistencylist,OGrougeprecisionlist)
print('Kendall Rank correlation b/w consistency and our rouge-l precision: {:.3f}'.format(corr3))
print('kendall Rank correlation b/w consistency and OG rouge-l precision: {:.3f}'.format(corr4))

corr5,_=kendalltau(fluencylist,rougePrecisionlist)
corr6,_=kendalltau(OGfluencylist,OGrougeprecisionlist)
print('kenda rank correlation b/w fluency and our rouge-l precision: {:.3f}'.format(corr5))
print('kenda rank correlation b/w fluency and OG rouge-l precision: {:.3f}'.format(corr6))

corr7,_=kendalltau(relevancelist,rougePrecisionlist)
corr8,_=kendalltau(OGrelevancelist,OGrougeprecisionlist)
print('kenda rank correlation b/w relevance and our rouge-l precision: {:.3f}'.format(corr7))
print('kenda rank correlation b/w relevance and OG rouge-l precision: {:.3f}'.format(corr8))




