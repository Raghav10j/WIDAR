import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
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

iq=0
Wredlist=[]
with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:

		print(iq)
		iq+=1
		

		for k in [0,1,2,3,4,5,6,7,8,9,10]:
			ref=obj['references'][k]
			article=obj['text']
			gold_summ_sent= six.ensure_str(ref).split(". ")
			article_Sent=six.ensure_str(article).split(". ")

			gold_summ_sent2=[]
			for line in gold_summ_sent:
			  if len(line)!=0:
			    gold_summ_sent2.append(line)



			article_Sent2=[]
			for line in article_Sent:
			  if len(line)!=0:
			    if line.isspace() is False:
			      article_Sent2.append(line.strip())


			wredsoftmax,wred=calculateWred(gold_summ_sent2)
			# print(mean(wred))

			Wredlist.append(mean(wred))





fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(Wredlist, bins = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],edgecolor='black')
 
# Show plot
plt.xlabel("Wred2")
plt.ylabel("y-axis")
plt.show()
plt.savefig('histograms/Wred2.png')
plt.savefig('histograms/Wred2.pdf')





