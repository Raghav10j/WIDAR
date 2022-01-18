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
from syntaticeval import weightedrouge

rougerecalllist=[]
rougefscorelist=[]
rougePrecisionlist=[]
coherencelist=[]
consistencylist=[]
fluencylist=[]
relevancelist=[]
with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		text=obj['text']
		decoded=obj['decoded']
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
		scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
		scores2 = scorer.score(text,
		                      decoded)


		recall=scores2['rouge1'][1]
		f_score=scores2['rouge1'][2]
		precision=scores2['rouge1'][0]
		rougerecalllist.append(recall)
		rougefscorelist.append(f_score)
		rougePrecisionlist.append(precision)



		# print(scores2['rouge1'])
		# break



corr1, _ = kendalltau(coherencelist,rougerecalllist)
print('Kendall Rank correlation with coherence and rouge-1 recall: {:.3f}'.format(corr1))
corr2, _ = kendalltau(coherencelist,rougefscorelist)
print('Kendall Rank correlation with coherence and rouge-1 fscore: {:.3f}'.format(corr2))
corr2, _ = kendalltau(coherencelist,rougePrecisionlist)
print('Kendall Rank correlation with coherence and rouge-1 precision: {:.3f}'.format(corr2))

print('-----------------')
corr3, _ = kendalltau(consistencylist,rougerecalllist)
print('Kendall Rank correlation with consistency and rouge-1 recall: {:.3f}'.format(corr3))
corr4, _ = kendalltau(consistencylist,rougefscorelist)
print('Kendall Rank correlation with consistency and rouge-1 fscore: {:.3f}'.format(corr4))
corr2, _ = kendalltau(consistencylist,rougePrecisionlist)
print('Kendall Rank correlation with consistency and rouge-1 precision: {:.3f}'.format(corr2))

print('--------------------')

corr5, _ = kendalltau(fluencylist,rougerecalllist)
print('Kendall Rank correlation with fluency and rouge-1 recall: {:.3f}'.format(corr5))
corr6, _ = kendalltau(fluencylist,rougefscorelist)
print('Kendall Rank correlation with fluency and rouge-1 fscore: {:.3f}'.format(corr6))
corr2, _ = kendalltau(fluencylist,rougePrecisionlist)
print('Kendall Rank correlation with fluency and rouge-1 precision: {:.3f}'.format(corr2))


print('---------------')

corr7, _ = kendalltau(relevancelist,rougerecalllist)
print('Kendall Rank correlation with relevance and rouge-2 recall: {:.3f}'.format(corr7))
corr8, _ = kendalltau(relevancelist,rougefscorelist)
print('Kendall Rank correlation with relevance and rouge-2 fscore: {:.3f}'.format(corr8))
corr2, _ = kendalltau(relevancelist,rougePrecisionlist)
print('Kendall Rank correlation with relevance and rouge-2 precision: {:.3f}'.format(corr2))

