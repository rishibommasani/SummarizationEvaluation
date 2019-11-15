import random
import torch
import math
# from newsroom import jsonl
import jsonlines as jsonl
from fragments import Fragments
from newsroom.analyze.rouge import ROUGE_N
from newsroom.analyze.rouge import ROUGE_L
import pickle
import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
spacy_tokenizer = spacy.load("en_core_web_sm")
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
path_to_mallet_binary = "Mallet/bin/mallet"
from transformers import BertTokenizer, BertForNextSentencePrediction
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from dataset import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
nlp = spacy.load('en', disable=['parser', 'ner'])
from fetch_data import * 
from converter import *

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): # Taken from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def load_datasets():
	# our_datasets = {'cnndm' : fetch_cnndm, 'nyt' : fetch_nyt , 'newsroom' : fetch_newsroom, 'tldr' : fetch_tldr, 'gigaword' : fetch_gigaword}
	# for dataset_name, fetcher in our_datasets.items():
	# 	data = fetcher()
	# 	our_statistics = compute_our_statistics(data, dataset_name)
	# 	# cmu_version = us2cmu(data)
	# 	data = []
	# 	#cmu_statistics = compute_cmu_statistics(cmu_version, dataset_name)
	# 	#cmu_version = []
	# 	print(dataset_name, our_statistics)

	cmu_datasets = fetch_cmu({'ami' : None, 'moviescript' : None, 'peerread' : None, 'pubmed' : None, 'xsum' : None})
	for dataset_name, cmu_version in cmu_datasets.items():
		#cmu_statistics = compute_cmu_statistics(cmu_version, dataset_name)
		data = cmu2us(cmu_version)
		#cmu_version = []
		our_statistics = compute_our_statistics(data, dataset_name)
		data = []
		print(dataset_name, our_statistics)


def compute_word_compression(data):
	print("Computing Word Compression")
	return 1 - (sum([1 / ex['compression'] for ex in data]) / len(data))


def compute_sentence_compression(data):
	print("Computing Sentence Compression")
	return 1 - (sum([len(sent_tokenize(ex['summary']))/len(sent_tokenize(ex['text'])) for ex in data]) / len(data))


def compute_ts(data, dataset, pickled=False):
	print("Computing Topic Similarity")
	k = 80
	p = 0.2
	N = 10
	n = 10000
	t = math.ceil(p * N)

	print("Parameters: k: {}, p: {}, N: {}, n: {}, t: {}".format(k, p, N, n, t))
	pickle_f = '{}.n={}.pickle'.format(dataset, n)

	if pickled:
		document_corpus, id2word_document, summary_corpus, id2word_summary = pickle.load(open(pickle_f, 'rb'))
	else:
		stop_words = stopwords.words('english')
		stop_words.extend(['from', 'subject', 're', 'edu', 'use', '>', '<', '/t', '-lrb-', '-rrb-', '/n', '\n'])
		data = data[:n]

		document_data = [[str(t).lower() for t in spacy_tokenizer(e['text']) if str(t).lower() not in stop_words] for e in data]
		document_data = lemmatization(document_data)
		id2word_document = corpora.Dictionary(document_data)
		document_corpus = [id2word_document.doc2bow(text) for text in document_data]

		summary_data = [[str(t).lower() for t in spacy_tokenizer(e['summary']) if str(t).lower() not in stop_words] for e in data]
		summary_data = lemmatization(summary_data)
		id2word_summary = corpora.Dictionary(summary_data)
		summary_corpus = [id2word_summary.doc2bow(text) for text in summary_data]

		pickle.dump((document_corpus, id2word_document, summary_corpus, id2word_summary), open(pickle_f, 'wb'))

	N_p_choices = [(N, p) for N in [8, 10] for p in [0.2, 0.35, 0.5]]
	for k in [160]:# [10, 20, 40, 80, 160]:
		print("Models being built have {} Topics".format(k))
		print("Computing Document Topic Model")
		lda_model_document = gensim.models.ldamodel.LdaModel(corpus=document_corpus,
		                                           id2word=id2word_document,
		                                           num_topics=k, 
		                                           random_state=100,
		                                           update_every=1,
		                                           chunksize=100,
		                                           passes=10,
		                                           alpha='auto',
		                                           per_word_topics=True)

		print("Computing Summary Topic Model")
		lda_model_summary = gensim.models.ldamodel.LdaModel(corpus=summary_corpus,
		                                           id2word=id2word_summary,
		                                           num_topics=k, 
		                                           random_state=100,
		                                           update_every=1,
		                                           chunksize=100,
		                                           passes=10,
		                                           alpha='auto',
		                                           per_word_topics=True)
		
		pickle.dump((lda_model_document, lda_model_summary), open("k={}.".format(k) + pickle_f, 'wb'))

		for N, p in N_p_choices:
			t = math.ceil(p * N)
			numerator = 0
			denominator = k
			for i in range(k):
				top_words_tuples = lda_model_summary.show_topic(i, N)
				top_words = {w for w, _ in top_words_tuples}
				for j in range(k):
					top_words_tuples_document = lda_model_document.show_topic(j, N)
					top_words_document = {w for w, _ in top_words_tuples_document}
					if len(top_words & top_words_document) >= t:
						numerator += 1
						break	
			print("Score: {}, N: {}, p: {}, k: {}".format(numerator / denominator, N, p, k))

	# numerator = 0
	# denominator = k
	# for i in range(k):
	# 	top_words_tuples = lda_model_summary.show_topic(i, N)
	# 	top_words = {w for w, _ in top_words_tuples}
	# 	for j in range(k):
	# 		top_words_tuples_document = lda_model_document.show_topic(j, N)
	# 		top_words_document = {w for w, _ in top_words_tuples_document}
	# 		if len(top_words & top_words_document) >= t:
	# 			numerator += 1
	# 			break			
	return numerator / denominator


def compute_abs1(data):
	print("Computing Abstractivity-1")
	return [x['coverage']/len(x) for x in data]
	return 1 - (sum([ex['coverage'] for ex in data]) / len(data))


def compute_abs2(data, dataset):
	print("Computing Abstractivity-2")
	output = []
	for ex in tqdm(data):
		if dataset == 'cnndm':
			output.append(ex['density'] / len([w for w in ex['summary'].split()]))
		else:
			output.append(ex['density'] / len(spacy_tokenizer(ex['summary'])))
	# return 1 - output/len(output)
	return 1 - (sum(output) / len(output))


def compute_red(data):
	print("Computing Redundancy")
	red1_output, red2_output, redL_output = [], [], []
	for ex in tqdm(data):
		summary = ex['summary']
		red1_scores, red2_scores, redL_scores = [], [], []
		sentences = sent_tokenize(summary)
		sentences = [" ".join([str(token).lower() for token in spacy_tokenizer(s)]) for s in sentences]
		if len(sentences) <= 1:
			red1_output.append(0)
			red2_output.append(0)
			redL_output.append(0)
		else:
			for i in range(len(sentences)):
				for j in range(i + 1, len(sentences)): # ROUGE is symmetric, so only do one of (a,b), (b,a)
					red1_scores.append(ROUGE_N(sentences[i], sentences[j], 1)[2]) # Rouge Triple of (p, r, f)
					red2_scores.append(ROUGE_N(sentences[i], sentences[j], 2)[2])
					redL_scores.append(ROUGE_L(sentences[i], sentences[j])[2])
			red1_output.append(max(red1_scores))
			red2_output.append(max(red2_scores))
			redL_output.append(max(redL_scores))
	assert len(red1_output) == len(data)
	assert len(red2_output) == len(data)
	assert len(redL_output) == len(data)
	return sum(red1_output) / len(red1_output), sum(red2_output) / len(red2_output), sum(redL_output) / len(redL_output)

# zip output with data and sort 
def compute_sc(data):
	print("Computing Semantic Coherence")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
	softmax = torch.nn.Softmax(dim=1)
	model.eval()
	output = []
	for ex in tqdm(data):
		summary = ex['summary']
		scores = []
		sentences = sent_tokenize(summary)
		if len(sentences) <= 1:
			output.append(1)
		else:
			numerator = 0
			denominator = len(sentences) - 1
			for i in range(len(sentences) - 1):
				prev = sentences[i]
				curr = sentences[i + 1]
				s = "[CLS] " + prev + " [SEP] " + curr + " [SEP]"
				tokenized_text = tokenizer.tokenize(s)
				boundary = tokenized_text.index('[SEP]')
				segment_ids = [0] * boundary + [1] * (len(tokenized_text) - boundary)
				indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				tokens_tensor = torch.tensor([indexed_tokens])
				segments_tensors = torch.tensor([segment_ids])
				with torch.no_grad():
					prediction = model(tokens_tensor, token_type_ids=segments_tensors)[0]
				prediction_sm = softmax(prediction)[0].tolist()
				if prediction_sm[0] > 0.5:
					numerator += 1
			output.append(numerator / denominator)
	print(len(data), len(output))
	# assert len(output) == len(data)
	return sum(output) / len(output)


def compute_our_statistics(data, dataset):
	print("Computing statistics for:", dataset)
	word_compression, sentence_compression, topic_similarity, abs1, abs2, red1, red2, redL, semantic_coherence = [None] * 9
	word_compression = compute_word_compression(data)
	sentence_compression = compute_sentence_compression(data)
	abs1 = compute_abs1(data)
	abs2 = compute_abs2(data, dataset)
	red1, red2, redL = compute_red(data)
	print({"CMP_W" : word_compression, "CMP_S" : sentence_compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL, "SC" : semantic_coherence})
	semantic_coherence = compute_sc(data)
	print({"CMP_W" : word_compression, "CMP_S" : sentence_compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL, "SC" : semantic_coherence})
	print()
	print()
	print()
	return {"CMP_W" : word_compression, "CMP_S" : sentence_compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL, "SC" : semantic_coherence}
	

if __name__ == '__main__':
	load_datasets()
	# cnndm_data = fetch_cnndm(pickled=True)
	# cnndm_data = cnndm_data[:20000]
	# newsroom_data = fetch_newsroom(pickled=True)
	# newsroom_data = newsroom_data[:20000]

	# tldr_data = fetch_tldr(pickled=False)
	# tldr_data = tldr_data[:20000]

	# note: can run add_sent_comp on data or just add it to when reading data
	# example: add_sent_comp(tldr_data)

	# gigaword_data = fetch_gigaword(pickled=True)
	# gigaword_data = gigaword_data[:20000]

	# nyt_data = fetch_nyt(pickled=True)
	
	# nyt_data = nyt_data[:20000]

	# exit()
	# statistics = compute_statistics(nyt_data, 'nyt')
	# print(statistics)``

	# return
	# statistics = compute_statistics(cnndm_data, 'cnndm')
	# print(statistics)
	# statistics = compute_statistics(newsroom_data, 'newsroom')
	# print(statistics)
	# statistics = compute_statistics(tldr_data, 'tldr')
	# print(statistics)
	# statistics = compute_statistics(gigaword_data, 'gigaword')
	# print(statistics)
	# return
	