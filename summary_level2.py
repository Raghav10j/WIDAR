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
from statistics import mean
from scipy.stats import kendalltau as correlator
from syntaticeval import *

test_id=[]


with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		summ_id=obj['id']
		if summ_id not in test_id:
			test_id.append(summ_id)

print(test_id)
print(len(test_id))



rougerecalllistsumm=[]
rougefscorelistsumm=[]
rougePrecisionlistsumm=[]
coherencelistsumm=[]
consistencylistsumm=[]
fluencylistsumm=[]
relevancelistsumm=[]

correlationCohwor=[]
correlationconswor=[]
correlationflwor=[]
correlationrelwor=[]

count=0
for summ_id in test_id:
	print(summ_id)
	print(count)
	count+=1

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
			if obj['id']==summ_id:

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
				for k in [0]:
					ref=obj['references'][k]
					article=obj['text']
					currrougetemprecall,currrougetempprecision,currrougetempfscore,x=weightedrouge(ref,decoded,article,'l')
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
		print(len(coherencelist))
		print(len(rougerecalllist))
		corr1,_=kendalltau(coherencelist,rougefscorelist)
		correlationCohwor.append(corr1)
		corr2,_=kendalltau(consistencylist,rougefscorelist)
		# print(len(consistencylist))
		# print(len(rougefscorelist))
		correlationconswor.append(corr2)
		# print(correlationconswor)
		corr3,_=kendalltau(fluencylist,rougefscorelist)
		correlationflwor.append(corr3)
		corr4,_=kendalltau(relevancelist,rougefscorelist)
		correlationrelwor.append(corr4)
		print(corr4)
		# rougerecalllistsumm.append(mean(rougerecalllist))
		# rougePrecisionlistsumm.append(mean(rougePrecisionlist))
		# rougefscorelistsumm.append(mean(rougefscorelistc))
		# coherencelistsumm.append(mean(coherencelist))
		# consistencylistsumm.append(mean(consistencylist))
		# fluencylistsumm.append(mean(fluencylist))
		# relevancelistsumm.append(mean(relevancelist))



OGrougerecalllistsumm=[]
OGrougeprecisionlistsumm=[]
OGrougefscorelistsumm=[]
OGcoherencelistsumm=[]
OGconsistencylistsumm=[]
OGfluencylistsumm=[]
OGrelevancelistsumm=[]
correlationCohwtr=[]
correlationconswtr=[]
correlationflwtr=[]
correlationrelwtr=[]


for summ_id in test_id:
	print(summ_id)


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
			if obj['model_id']!='M23_dynamicmix' and obj['id']==summ_id:
				print(iq)
				iq=iq+1
				OGrougerecalllist.append(obj['metric_scores_1']['rouge']['rouge_l_recall'])
				OGrougeprecisionlist.append(obj['metric_scores_1']['rouge']['rouge_l_precision'])
				OGrougefscorelist.append(obj['metric_scores_1']['rouge']['rouge_l_f_score'])
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
		print(len(OGcoherencelist))
		print(len(OGrougerecalllist))
		corr1,_=kendalltau(OGcoherencelist,OGrougefscorelist)
		correlationCohwtr.append(corr1)
		corr2,_=kendalltau(OGconsistencylist,OGrougefscorelist)
		correlationconswtr.append(corr2)
		corr3,_=kendalltau(OGfluencylist,OGrougefscorelist)
		correlationflwtr.append(corr3)
		corr4,_=kendalltau(OGrelevancelist,OGrougefscorelist)
		correlationrelwtr.append(corr4)

		# OGrougerecalllistsumm.append(mean(OGrougerecalllist))
		# OGrougeprecisionlistsumm.append(mean(OGrougeprecisionlist))
		# OGrougefscorelistsumm.appedn(mean(OGrougefscorelist))
		# OGcoherencelistsumm.append(OGcoherencelist)
		# OGconsistencylistsumm.append(OGconsistencylist)
		# OGfluencylistsumm.append(OGfluencylist)
		# OGrelevancelistsumm.append(OGrelevancelist)

print(len(correlationCohwor))
print(len(correlationCohwtr))
print('correlation b/w coherence and our weighted rouge1 recall : {}'.format(mean(correlationCohwor)))
print('correlation b/w coherence and their rouge1 recall : {}'.format(mean(correlationCohwtr)))

print('correlation b/w consistency and our weighted rouge1 recall : {}'.format(mean(correlationconswor)))
print('correlation b/w consistency and their rouge1 recall : {}'.format(mean(correlationconswtr)))

print('correlation b/w fluency and our weighted rouge1 recall : {}'.format(mean(correlationflwor)))
print('correlation b/w fluency and their rouge1 recall : {}'.format(mean(correlationflwtr)))

print('correlation b/w relevance and our weighted rouge1 recall : {}'.format(mean(correlationrelwor)))
print('correlation b/w relevance and their rouge1 recall : {}'.format(mean(correlationrelwtr)))


# # print('----------------without Wred-----------------------')

# print('----------------recall---------------')
# corr1,_=kendalltau(coherencelist,rougerecalllist)
# corr2,_=kendalltau(OGcoherencelist,OGrougerecalllist)
# print('Kendall Rank correlation b/w coherence and our rouge-2 recall: {:.3f}'.format(corr1))
# print('Kendall Rank correlation b/w coherence and OG rouge-2 recall: {:.3f}'.format(corr2))

# corr3,_=kendalltau(consistencylist,rougerecalllist)
# corr4,_=kendalltau(OGconsistencylist,OGrougerecalllist)
# print('Kendall Rank correlation b/w consistency and our rouge-2 recall: {:.3f}'.format(corr3))
# print('kendall Rank correlation b/w consistency and OG rouge-2 recall: {:.3f}'.format(corr4))

# corr5,_=kendalltau(fluencylist,rougerecalllist)
# corr6,_=kendalltau(OGfluencylist,OGrougerecalllist)
# print('kenda rank correlation b/w fluency and our rouge-2 recall: {:.3f}'.format(corr5))
# print('kenda rank correlation b/w fluency and OG rouge-2 recall: {:.3f}'.format(corr6))

# corr7,_=kendalltau(relevancelist,rougerecalllist)
# corr8,_=kendalltau(OGrelevancelist,OGrougerecalllist)
# print('kenda rank correlation b/w relevance and our rouge-2 recall: {:.3f}'.format(corr7))
# print('kenda rank correlation b/w relevance and OG rouge-2 recall: {:.3f}'.format(corr8))

# print('----------------fscore---------------')
# corr1,_=kendalltau(coherencelist,rougefscorelist)
# corr2,_=kendalltau(OGcoherencelist,OGrougefscorelist)
# print('Kendall Rank correlation b/w coherence and our rouge-2 fscore: {:.3f}'.format(corr1))
# print('Kendall Rank correlation b/w coherence and OG rouge-2 fscore: {:.3f}'.format(corr2))

# corr3,_=kendalltau(consistencylist,rougefscorelist)
# corr4,_=kendalltau(OGconsistencylist,OGrougefscorelist)
# print('Kendall Rank correlation b/w consistency and our rouge-2 fscore: {:.3f}'.format(corr3))
# print('kendall Rank correlation b/w consistency and OG rouge-2 fscore: {:.3f}'.format(corr4))

# corr5,_=kendalltau(fluencylist,rougefscorelist)
# corr6,_=kendalltau(OGfluencylist,OGrougefscorelist)
# print('kenda rank correlation b/w fluency and our rouge-2 fscore: {:.3f}'.format(corr5))
# print('kenda rank correlation b/w fluency and OG rouge-2 fscore: {:.3f}'.format(corr6))

# corr7,_=kendalltau(relevancelist,rougefscorelist)
# corr8,_=kendalltau(OGrelevancelist,OGrougefscorelist)
# print('kenda rank correlation b/w relevance and our rouge-2 fscore: {:.3f}'.format(corr7))
# print('kenda rank correlation b/w relevance and OG rouge-2 fscore: {:.3f}'.format(corr8))



# print('----------------precision---------------')
# corr1,_=kendalltau(coherencelist,rougePrecisionlist)
# corr2,_=kendalltau(OGcoherencelist,OGrougeprecisionlist)
# print('Kendall Rank correlation b/w coherence and our rouge-2 precision: {:.3f}'.format(corr1))
# print('Kendall Rank correlation b/w coherence and OG rouge-2 precision: {:.3f}'.format(corr2))

# corr3,_=kendalltau(consistencylist,rougePrecisionlist)
# corr4,_=kendalltau(OGconsistencylist,OGrougeprecisionlist)
# print('Kendall Rank correlation b/w consistency and our rouge-2 precision: {:.3f}'.format(corr3))
# print('kendall Rank correlation b/w consistency and OG rouge-2 precision: {:.3f}'.format(corr4))

# corr5,_=kendalltau(fluencylist,rougePrecisionlist)
# corr6,_=kendalltau(OGfluencylist,OGrougeprecisionlist)
# print('kenda rank correlation b/w fluency and our rouge-2 precision: {:.3f}'.format(corr5))
# print('kenda rank correlation b/w fluency and OG rouge-2 precision: {:.3f}'.format(corr6))

# corr7,_=kendalltau(relevancelist,rougePrecisionlist)
# corr8,_=kendalltau(OGrelevancelist,OGrougeprecisionlist)
# print('kenda rank correlation b/w relevance and our rouge-2 precision: {:.3f}'.format(corr7))
# print('kenda rank correlation b/w relevance and OG rouge-2 precision: {:.3f}'.format(corr8))




