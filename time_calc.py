import json
import jsonlines
import pickle
import six
import time

# from eval import compute_rouge_n,tokenizef,_summary_level_lcs,sentencelevelrougeL

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau,pearsonr
from rouge_score import rouge_scorer
import statistics
from scipy.stats import kendalltau as correlator
from syntaticeval import *
from statistics import mean



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
		if obj['model_id']!='p':
			print(iq)
			iq=iq+1
			OGrougerecalllist.append(obj['metric_scores_11']['rouge']['rouge_2_recall'])
			OGrougeprecisionlist.append(obj['metric_scores_11']['rouge']['rouge_2_precision'])
			OGrougefscorelist.append(obj['metric_scores_11']['rouge']['rouge_2_f_score'])
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



# rougerecalllist=[]
# rougefscorelist=[]
# rougePrecisionlist=[]
# coherencelist=[]
# consistencylist=[]
# fluencylist=[]
# relevancelist=[]


# iq=0

# # with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
# # 	for obj in reader:
# # 		text=obj['text']
# # 		decoded=obj['decoded']
# # 		print(iq)
# # 		iq+=1
# # 		t1=obj['expert_annotations'][0]
# # 		t2=obj['expert_annotations'][1]
# # 		t3=obj['expert_annotations'][2]
# # 		# consistency': 1, 'fluency': 4, 'relevance': 2
# # 		coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
# # 		consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
# # 		fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
# # 		relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
# # 		coherencelist.append(coherenceval)
# # 		consistencylist.append(consistencyval)
# # 		fluencylist.append(fluencyval)
# # 		relevancelist.append(relevanceval)

# # 		currrougerecall=0
# # 		currrougeprecision=0
# # 		currrougefscore=0
# # 		for k in [0]:
# # 			ref=obj['references'][k]
# # 			article=obj['text']
# # 			currrougetemprecall,currrougetempprecision,currrougetempfscore,wcov=weightedrouge(ref,decoded,article,2)
# # 			# currrougetemprecall,currrougetempprecision,currrougetempfscore=0,0,0
# # 			scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# # 			scores2 = scorer.score(text,
# # 			                      decoded)


# # 			f_score=scores2['rougeL'][2]
# # 			# gold_summ_sent= six.ensure_str(ref).split(". ")
# # 			# article_Sent=six.ensure_str(article).split(". ")

# # 			# gold_summ_sent2=[]
# # 			# for line in gold_summ_sent:
# # 			#   if len(line)!=0:
# # 			#     gold_summ_sent2.append(line)



# # 			# article_Sent2=[]
# # 			# for line in article_Sent:
# # 			#   if len(line)!=0:
# # 			#     if line.isspace() is False:
# # 			#       article_Sent2.append(line.strip())


# # 			# wcovsoftmax,wcov=calculateWcov(gold_summ_sent2,article_Sent2)
# # 			lamda=0.5
# # 			# print(lamda)
# # 			# lamda=0
# # 			# print(lamda)
# # 			currfinalrougetemprecall=(((1-lamda)*f_score)+(lamda*currrougetemprecall))
# # 			currfinalrougefscore=(((1-lamda)*f_score)+(lamda*currrougetempfscore))
# # 			currfinalrougeprecision=(((1-lamda)*f_score)+(lamda*currrougetempprecision))



# # 			currrougerecall=currrougerecall+currfinalrougetemprecall
# # 			currrougeprecision=currrougeprecision+currfinalrougeprecision
# # 			currrougefscore=currrougefscore+currfinalrougefscore
# # 			# currrouge+=currrougetemp
# # 		currrougerecall=currrougerecall/1
# # 		rougerecalllist.append(currrougerecall)
# # 		currrougeprecision=currrougeprecision/1
# # 		rougePrecisionlist.append(currrougeprecision)
# # 		currrougefscore=currrougefscore/1
# # 		rougefscorelist.append(currrougefscore)
# # 		# rougelist.append(currrouge)


# start_time = time.time()

# with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
# 	for obj in reader:
# 		text=obj['text']
# 		decoded=obj['decoded']
# 		print(iq)
# 		iq+=1
# 		t1=obj['expert_annotations'][0]
# 		t2=obj['expert_annotations'][1]
# 		t3=obj['expert_annotations'][2]
# 		# consistency': 1, 'fluency': 4, 'relevance': 2
# 		coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
# 		consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
# 		fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
# 		relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
# 		coherencelist.append(coherenceval)
# 		consistencylist.append(consistencyval)
# 		fluencylist.append(fluencyval)
# 		relevancelist.append(relevanceval)

# 		currrougerecall=0
# 		currrougeprecision=0
# 		currrougefscore=0
# 		for k in [0]:
# 			ref=obj['references'][k]
# 			article=obj['text']
# 			# currrougetemprecall,currrougetempprecision,currrougetempfscore,wcov=weightedrouge(ref,decoded,article,2)
# 			# currrougetemprecall,currrougetempprecision,currrougetempfscore=0,0,0
# 			scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# 			scores2 = scorer.score(ref,
# 			                      decoded)


# 			f_score=scores2['rougeL'][2]
# 			# gold_summ_sent= six.ensure_str(ref).split(". ")
# 			# article_Sent=six.ensure_str(article).split(". ")

# 			# gold_summ_sent2=[]
# 			# for line in gold_summ_sent:
# 			#   if len(line)!=0:
# 			#     gold_summ_sent2.append(line)



# 			# article_Sent2=[]
# 			# for line in article_Sent:
# 			#   if len(line)!=0:
# 			#     if line.isspace() is False:
# 			#       article_Sent2.append(line.strip())


# 			# wcovsoftmax,wcov=calculateWcov(gold_summ_sent2,article_Sent2)
# 		# 	lamda=0.5
# 		# 	# print(lamda)
# 		# 	# lamda=0
# 		# 	# print(lamda)
# 		# 	currfinalrougetemprecall=(((1-lamda)*f_score)+(lamda*currrougetemprecall))
# 		# 	currfinalrougefscore=(((1-lamda)*f_score)+(lamda*currrougetempfscore))
# 		# 	currfinalrougeprecision=(((1-lamda)*f_score)+(lamda*currrougetempprecision))



# 		# 	currrougerecall=currrougerecall+currfinalrougetemprecall
# 		# 	currrougeprecision=currrougeprecision+currfinalrougeprecision
# 		# 	currrougefscore=currrougefscore+currfinalrougefscore
# 		# 	# currrouge+=currrougetemp
# 		# currrougerecall=currrougerecall/1
# 		# rougerecalllist.append(currrougerecall)
# 		# currrougeprecision=currrougeprecision/1
# 		# rougePrecisionlist.append(currrougeprecision)
# 		# currrougefscore=currrougefscore/1
# 		# rougefscorelist.append(currrougefscore)
# 		# rougelist.append(currrouge)

# print("--- %s seconds ---" % (time.time() - start_time))
# time.sleep(60)


# # iq=0
# # Wcovlist=[]
# # with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
# # 	for obj in reader:

# # 		print(iq)
# # 		iq+=1
		

# # 		for k in [0,1,2,3,4,5,6,7,8,9,10]:
# 			ref=obj['references'][k]
# 			article=obj['text']
# 			gold_summ_sent= six.ensure_str(ref).split(". ")
# 			article_Sent=six.ensure_str(article).split(". ")

# 			gold_summ_sent2=[]
# 			for line in gold_summ_sent:
# 			  if len(line)!=0:
# 			    gold_summ_sent2.append(line)



# 			article_Sent2=[]
# 			for line in article_Sent:
# 			  if len(line)!=0:
# 			    if line.isspace() is False:
# 			      article_Sent2.append(line.strip())


# 			wcovsoftmax,wcov=calculateWcov(gold_summ_sent2,article_Sent2)
# 			# print(mean(wred))

# 			Wcovlist.append(mean(wcov))




# articlerougerecalllist=[]
# articlerougefscorelist=[]
# articlerougePrecisionlist=[]
# iq=0
# with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
# 	for obj in reader:
# 		print(iq)
# 		iq+=1
# 		text=obj['text']
# 		decoded=obj['decoded']
# 		# t1=obj['expert_annotations'][0]
# 		# t2=obj['expert_annotations'][1]
# 		# t3=obj['expert_annotations'][2]
# 		# # consistency': 1, 'fluency': 4, 'relevance': 2
# 		# coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
# 		# consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
# 		# fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
# 		# relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
# 		# coherencelist.append(coherenceval)
# 		# consistencylist.append(consistencyval)
# 		# fluencylist.append(fluencyval)
# 		# relevancelist.append(relevanceval)
# 		scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# 		scores2 = scorer.score(text,
# 		                      decoded)


# 		recall=scores2['rougeL'][1]
# 		f_score=scores2['rougeL'][2]
# 		precision=scores2['rougeL'][0]
# 		articlerougerecalllist.append(recall)
# 		articlerougefscorelist.append(f_score)
# 		articlerougePrecisionlist.append(precision)


# finalevalrecall=[]
# finalevalprecision=[]
# finalevalfscore=[]

# for i in range(len(articlerougefscorelist)):
# 	finalevalfscore.append(((0.5*articlerougefscorelist[i])+(0.5*rougefscorelist[i])))
# 	finalevalprecision.append(((0.5*articlerougefscorelist[i])+(0.5*rougePrecisionlist[i])))
# 	finalevalrecall.append(((0.5*articlerougefscorelist[i])+(0.5*rougerecalllist[i])))








# print('----------------without Wred-----------------------')
# print("--- %s seconds ---" % (time.time() - start_time))
print('----------------recall---------------')
# corr1,_=kendalltau(coherencelist,rougerecalllist)
corr2,_=pearsonr(OGcoherencelist,OGrougerecalllist)
# print('Kendall Rank correlation b/w coherence and our final rouge-l recall: {:.3f}'.format(corr1))
print('Kendall Rank correlation b/w coherence and OG rouge-l recall: {:.3f}'.format(corr2))

# corr3,_=kendalltau(consistencylist,rougerecalllist)
corr4,_=pearsonr(OGconsistencylist,OGrougerecalllist)
# print('Kendall Rank correlation b/w consistency and our final rouge-l recall: {:.3f}'.format(corr3))
print('kendall Rank correlation b/w consistency and OG rouge-l recall: {:.3f}'.format(corr4))

# corr5,_=kendalltau(fluencylist,rougerecalllist)
corr6,_=pearsonr(OGfluencylist,OGrougerecalllist)
# print('kenda rank correlation b/w fluency and our final rouge-l recall: {:.3f}'.format(corr5))
print('kenda rank correlation b/w fluency and OG rouge-l recall: {:.3f}'.format(corr6))

# corr7,_=kendalltau(relevancelist,rougerecalllist)
corr8,_=pearsonr(OGrelevancelist,OGrougerecalllist)
# print('kenda rank correlation b/w relevance and our final rouge-l recall: {:.3f}'.format(corr7))
print('kenda rank correlation b/w relevance and OG rouge-l recall: {:.3f}'.format(corr8))

print('----------------fscore---------------')
# corr1,_=kendalltau(coherencelist,rougefscorelist)
corr2,_=pearsonr(OGcoherencelist,OGrougefscorelist)
# print('Kendall Rank correlation b/w coherence and our final rouge-l fscore: {:.3f}'.format(corr1))
print('Kendall Rank correlation b/w coherence and OG rouge-l fscore: {:.3f}'.format(corr2))

# corr3,_=kendalltau(consistencylist,rougefscorelist)
corr4,_=pearsonr(OGconsistencylist,OGrougefscorelist)
# print('Kendall Rank correlation b/w consistency and our final rouge-l fscore: {:.3f}'.format(corr3))
print('kendall Rank correlation b/w consistency and OG rouge-l fscore: {:.3f}'.format(corr4))

# corr5,_=kendalltau(fluencylist,rougefscorelist)
corr6,_=pearsonr(OGfluencylist,OGrougefscorelist)
# print('kenda rank correlation b/w fluency and our final rouge-l fscore: {:.3f}'.format(corr5))
print('kenda rank correlation b/w fluency and OG rouge-l fscore: {:.3f}'.format(corr6))

# corr7,_=kendalltau(relevancelist,rougefscorelist)
corr8,_=pearsonr(OGrelevancelist,OGrougefscorelist)
# print('kenda rank correlation b/w relevance and our final rouge-l fscore: {:.3f}'.format(corr7))
print('kenda rank correlation b/w relevance and OG rouge-l fscore: {:.3f}'.format(corr8))



# print('----------------precision---------------')
# corr1,_=kendalltau(coherencelist,rougePrecisionlist)
# corr2,_=kendalltau(OGcoherencelist,OGrougeprecisionlist)
# print('Kendall Rank correlation b/w coherence and our final rouge-l precision: {:.3f}'.format(corr1))
# print('Kendall Rank correlation b/w coherence and OG rouge-l precision: {:.3f}'.format(corr2))

# corr3,_=kendalltau(consistencylist,rougePrecisionlist)
# corr4,_=kendalltau(OGconsistencylist,OGrougeprecisionlist)
# print('Kendall Rank correlation b/w consistency and our final rouge-l precision: {:.3f}'.format(corr3))
# print('kendall Rank correlation b/w consistency and OG rouge-l precision: {:.3f}'.format(corr4))

# corr5,_=kendalltau(fluencylist,rougePrecisionlist)
# corr6,_=kendalltau(OGfluencylist,OGrougeprecisionlist)
# print('kenda rank correlation b/w fluency and our final rouge-l precision: {:.3f}'.format(corr5))
# print('kenda rank correlation b/w fluency and OG rouge-l precision: {:.3f}'.format(corr6))

# corr7,_=kendalltau(relevancelist,rougePrecisionlist)
# corr8,_=kendalltau(OGrelevancelist,OGrougeprecisionlist)
# print('kenda rank correlation b/w relevance and our final rouge-l precision: {:.3f}'.format(corr7))
# print('kenda rank correlation b/w relevance and OG rouge-l precision: {:.3f}'.format(corr8))




# Kendall Rank correlation b/w coherence and our final rouge-l fscore: 0.132
# Kendall Rank correlation b/w coherence and OG rouge-l fscore: 0.109
# Kendall Rank correlation b/w consistency and our final rouge-l fscore: 0.132
# kendall Rank correlation b/w consistency and OG rouge-l fscore: 0.090
# kenda rank correlation b/w fluency and our final rouge-l fscore: 0.097
# kenda rank correlation b/w fluency and OG rouge-l fscore: 0.067
# kenda rank correlation b/w relevance and our final rouge-l fscore: 0.238
# kenda rank correlation b/w relevance and OG rouge-l fscore: 0.216
