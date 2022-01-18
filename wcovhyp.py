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




corrcoherence=[]
corrfluency=[]
corrconsistency=[]
corrrelevance=[]

for th in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
	print('threshold for wcov : {}'.format(th))




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
			text=obj['text']
			decoded=obj['decoded']
			print('{} : {} '.format(th,iq))
			# print(iq)
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

				currrougetemprecall,currrougetempprecision,currrougetempfscore,wcov=weightedrouge(ref,decoded,article,'l')
				

				# currrougetemprecall,currrougetempprecision,currrougetempfscore=0,0,0


				lamda=1

				currfinalrougetemprecall=(lamda*currrougetemprecall)
				currfinalrougefscore=(lamda*currrougetempfscore)
				currfinalrougeprecision=(lamda*currrougetempprecision)



				currrougerecall=currrougerecall+currfinalrougetemprecall
				currrougeprecision=currrougeprecision+currfinalrougeprecision
				currrougefscore=currrougefscore+currfinalrougefscore
				# currrouge+=currrougetemp
			currrougerecall=currrougerecall/11
			rougerecalllist.append(currrougerecall)
			currrougeprecision=currrougeprecision/11
			rougePrecisionlist.append(currrougeprecision)
			currrougefscore=currrougefscore/11
			rougefscorelist.append(currrougefscore)





	





	# print('----------------without Wred-----------------------')

	# print('----------------recall---------------')
	# corr1,_=kendalltau(coherencelist,rougerecalllist)
	# corr2,_=kendalltau(OGcoherencelist,OGrougerecalllist)
	# print('Kendall Rank correlation b/w coherence and our final rouge-l recall: {:.3f}'.format(corr1))
	# print('Kendall Rank correlation b/w coherence and OG rouge-l recall: {:.3f}'.format(corr2))

	# corr3,_=kendalltau(consistencylist,rougerecalllist)
	# corr4,_=kendalltau(OGconsistencylist,OGrougerecalllist)
	# print('Kendall Rank correlation b/w consistency and our final rouge-l recall: {:.3f}'.format(corr3))
	# print('kendall Rank correlation b/w consistency and OG rouge-l recall: {:.3f}'.format(corr4))

	# corr5,_=kendalltau(fluencylist,rougerecalllist)
	# corr6,_=kendalltau(OGfluencylist,OGrougerecalllist)
	# print('kenda rank correlation b/w fluency and our final rouge-l recall: {:.3f}'.format(corr5))
	# print('kenda rank correlation b/w fluency and OG rouge-l recall: {:.3f}'.format(corr6))

	# corr7,_=kendalltau(relevancelist,rougerecalllist)
	# corr8,_=kendalltau(OGrelevancelist,OGrougerecalllist)
	# print('kenda rank correlation b/w relevance and our final rouge-l recall: {:.3f}'.format(corr7))
	# print('kenda rank correlation b/w relevance and OG rouge-l recall: {:.3f}'.format(corr8))

	print('----------------fscore---------------')
	corr1,_=kendalltau(coherencelist,rougefscorelist)
	corrcoherence.append(corr1)

	print('Kendall Rank correlation b/w coherence and our final rouge-l fscore: {:.3f}'.format(corr1))

	corr3,_=kendalltau(consistencylist,rougefscorelist)
	corrconsistency.append(corr3)
	print('Kendall Rank correlation b/w consistency and our final rouge-l fscore: {:.3f}'.format(corr3))

	corr5,_=kendalltau(fluencylist,rougefscorelist)
	corrfluency.append(corr5)
	print('kenda rank correlation b/w fluency and our final rouge-l fscore: {:.3f}'.format(corr5))

	corr7,_=kendalltau(relevancelist,rougefscorelist)
	corrrelevance.append(corr7)
	print('kenda rank correlation b/w relevance and our final rouge-l fscore: {:.3f}'.format(corr7))

	time.sleep(120)




with open("weightesranges/coherencediffwcov3F1L.txt", "wb") as fp:
	pickle.dump(corrcoherence, fp)

with open("weightesranges/consistencydiffwcov3F1L.txt", "wb") as fp:
	pickle.dump(corrconsistency, fp)


with open("weightesranges/fluencydiffwcov3F1L.txt", "wb") as fp:
	pickle.dump(corrfluency, fp)

with open("weightesranges/relevancediffwcov3F1L.txt", "wb") as fp:
	pickle.dump(corrrelevance, fp)

