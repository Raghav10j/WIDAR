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
from syntaticeval import *

def flatten(t):

    return [item for sublist in t for item in sublist]


# def calculateWred(references):
#   sim=[]

#   if len(references)==1:
#     # print('------')
#     sim=[1]

#     return sim



#   for curridx in range(len(references)):

#     currwred=0
#     for otheridx in range(len(references)):
#       if curridx!=otheridx:
#         currsen=references[curridx]
#         othersen=references[otheridx]
#         # rouge = Rouge()
#         # scores = rouge.get_scores(othersen, currsen)
#         # similarity=scores[0]['rouge-1']['r']
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#         scores2 = scorer.score(currsen,
#                               othersen)
#         similarity=scores2['rougeL'][1]
#         sim.append(similarity)


#   return sim


def calculateWcov(references,article):
  sim=[]

  for lines1 in references:

    currwcov=0
    for lines2 in article:

      # rouge = Rouge()
      # scores = rouge.get_scores(lines1, lines2)
      # similarity=scores[0]['rouge-1']['r']
      scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
      scores2 = scorer.score(lines2,
                            lines1)
      similarity=scores2['rougeL'][1]
      sim.append(similarity)




  return sim



iq=0
allcovvalues=[]

with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		# text=obj['text']
		decoded=obj['decoded']
		print(iq)
		iq+=1


		currrougerecall=0
		currrougeprecision=0
		currrougefscore=0
		for k in [0,1,2,3,4,5,6,7,8,9,10]:
			ref=obj['references'][k]
			article=obj['text']
			article_Sent=six.ensure_str(article).split(". ")
			article_Sent2=[]
			for line in article_Sent:
			  if len(line)!=0:
			    if line.isspace() is False:
			      article_Sent2.append(line.strip())
			gold_summ_sent= six.ensure_str(ref).split(". ")
			gold_summ_sent2=[]
			for line in gold_summ_sent:
			  if len(line)!=0:
			    gold_summ_sent2.append(line)

			sim=calculateWcov(gold_summ_sent2,article_Sent2)
			allcovvalues.append(sim)





wcovrange=flatten(allcovvalues)


with open("weightesranges/wcov.txt", "wb") as fp:
	pickle.dump(wcovrange, fp)









