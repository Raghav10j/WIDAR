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
with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		if obj['model_id']=='M8':
		# text=obj['text']
			decoded=obj['decoded']
			print('decoded Summary:')
			print(decoded)

			for k in [0]:
				ref=obj['references'][k]
				article=obj['text']
				print('ref')
				print(ref)
				print('article')
				print(article)

			val = input("Enter ")


