from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cytoolz import concat, curry
import re
import six
import itertools
import statistics
import pickle

import collections


from collections import Counter
from gensim.models import KeyedVectors as kv
from rouge import Rouge 
from scipy.special import softmax

import nltk
nltk.download('punkt')
from nltk import tokenize
from nltk import download
from nltk.corpus import stopwords
import numpy as np
download('stopwords')



from collections import Counter
from gensim.models import KeyedVectors as kv

import nltk
#nltk.download('punkt')
from nltk import tokenize
from nltk import download
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stem = SnowballStemmer("english")
import numpy as np

# from scipy.stats import pearsonr

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
#download('stopwords')
from rouge_score import rouge_scorer


def tokenizef(text, stemmer):
  """Tokenize input text into a list of tokens.
  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.
  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
  Returns:
    A list of string tokens extracted from input text.
  """

  # Convert everything to lowercase.
  text = text.lower()
  # Replace any non-alpha-numeric characters with spaces.
  text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

  tokens = re.split(r"\s+", text)
  if stemmer:
    # Only stem words more than 3 characters long.
    tokens = [stem.stem(x) if len(x) > 3 else x for x in tokens]

  # One final check to drop any empty or invalid tokens.
  tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

  return tokens


def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    # print(*ngrams)
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split() for _ in sentences]))





def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs(a, b):

    """ compute the longest common subsequence between a and b"""

    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = collections.deque()
    while (i > 0 and j > 0):

        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs

def compute_rouge_l_summ(summs, refs, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    tot_hit = 0
    print(refs)
    ref_cnt = Counter(concat(refs))
    print(ref_cnt)
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        print(ref)
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum((len(s) for s in summs))
        recall = tot_hit / sum((len(r) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score







def _lcs_table(ref, can):
  """Create 2-d LCS score table."""
  rows = len(ref)
  cols = len(can)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if ref[i - 1] == can[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def _backtrack_norec(t, ref, can):
  """Read out LCS."""
  i = len(ref)
  j = len(can)
  lcs = []
  while i > 0 and j > 0:
    if ref[i - 1] == can[j - 1]:
      lcs.insert(0, i-1)
      i -= 1
      j -= 1
    elif t[i][j - 1] > t[i - 1][j]:
      j -= 1
    else:
      i -= 1
  return lcs


def _summary_level_lcs(ref_sent, can_sent):
  """ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.
  Args:
    ref_sent: list of tokenized reference sentences
    can_sent: list of tokenized candidate sentences
  Returns:
    summary level ROUGE score
  """
  if not ref_sent or not can_sent:
    return 0

  m = sum(map(len, ref_sent))

  n = sum(map(len, can_sent))
  if not n or not m:
    return 0

  # get token counts to prevent double counting
  token_cnts_r = collections.Counter()
  token_cnts_c = collections.Counter()
  for s in ref_sent:
    # s is a list of tokens
    token_cnts_r.update(s)
  for s in can_sent:
    token_cnts_c.update(s)

  hits = 0
  i=0
  for r in ref_sent:
    lcs = _union_lcs(r, can_sent)
    # print(r)
    # print('LCS: {}'.format(lcs))

    # hits=hits+len(_union_lcs(r,can_sent))
    # Prevent double-counting:
    # The paper describes just computing hits += len(_union_lcs()),
    # but the implementation prevents double counting. We also
    # implement this as in version 1.5.5.
    for t in lcs:
      if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
        hits = hits+1
        # print('weight for {} is {}'.format(i,weights[i]))

        token_cnts_c[t] -= 1
        token_cnts_r[t] -= 1
    i=i+1

  recall = hits / m
  precision = hits / n
  if recall!=0 and precision!=0:
    fscore=2*((precision*recall)/(precision+recall))
  else:
    fscore=0
  # fmeasure = scoring.fmeasure(precision, recall)
  return recall,precision


def _union_lcs(ref, c_list):
  """Find union LCS between a ref sentence and list of candidate sentences.
  Args:
    ref: list of tokens
    c_list: list of list of indices for LCS into reference summary
  Returns:
    List of tokens in ref representing union LCS.
  """
  lcs_list = [lcs_ind(ref, c) for c in c_list]
  return [ref[i] for i in _find_union(lcs_list)]


def _find_union(lcs_list):
  """Finds union LCS given a list of LCS."""
  return sorted(list(set().union(*lcs_list)))


def lcs_ind(ref, can):
  """Returns one of the longest lcs."""
  t = _lcs_table(ref, can)        # N        # Note: Does not support multi-line text.ote: Does not support multi-line text.

  return _backtrack_norec(t, ref, can)


def sentencelevelrougeL(gold_summ,predicted_summ):
    predicted_summ_sent = six.ensure_str(predicted_summ).split(". ")
    predicted_summ_sent2=[]
    for s in predicted_summ_sent:
      if len(s)!=0:
        predicted_summ_sent2.append(tokenizef(s,True))


    gold_summ_sent= six.ensure_str(gold_summ).split(". ")
    gold_summ_sent2=[]
    for s in gold_summ_sent:
      if len(s)!=0:

        gold_summ_sent2.append(tokenizef(s,True))




    recall,precision = _summary_level_lcs(gold_summ_sent2,predicted_summ_sent2)
    if precision==0 and recall==0:
        fmeasure=0
    else:
        fmeasure=2*((precision*recall)/(precision+recall))
    return recall,precision,fmeasure

# rougescore=[]

# for i in range(0,100):
#   curridx=i
#   idealidx=''
#   idealidx=idealidx+str(curridx)
#   rempos=6-len(idealidx)
#   for i in range(rempos):
#     idealidx='0'+idealidx

#   print(idealidx)
#   decoded_add='/Data/anubhavcs17/asurl/summeval/SummEval/M8decoded/'+idealidx+'_decoded.txt'
#   file2 = open(decoded_add,'r')
#   decoded=file2.read()

#   currrouge=0

#   for k in [0,1,2,3,4,5,6,7,8,9,10]:
#     print(k)
#     ref_add='/Data/anubhavcs17/asurl/summeval/SummEval/M8reference2/'+idealidx+'_reference'+str(k)+'.txt'
#     file1 = open(ref_add,"r")
#     ref=file1.read()
#     rougecurr=compute_rouge_n(tokenizef(decoded,True),tokenizef(ref,True),1,'f')
#     # scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#     # scores2 = scorer.score(ref,
#     #                       decoded)
#     # rougecurr=scores2['rouge1'][1]

#     currrouge=max(rougecurr,currrouge)


#   rougescore.append(currrouge)


# avgrouge=statistics.mean(rougescore)

# with open("/Data/anubhavcs17/asurl/summeval/SummEval/coherenceM8.txt", "rb") as fp:
#   coherence = pickle.load(fp)

# # print(len(coherence))
# # print(coherence[5])
# # print(coherence)




# corr1, _ = pearsonr(coherence, rougescore)
# print('Pearsons correlation with coherence: {:.3f}'.format(corr1))
# corr2, _ = spearmanr(coherence, rougescore)
# print('Spearmans correlation with coherence: {:.3f}'.format(corr2))
# corr3, _ = kendalltau(coherence, rougescore)
# print('Kendall Rank correlation with coherence: {:.3f}'.format(corr3))




