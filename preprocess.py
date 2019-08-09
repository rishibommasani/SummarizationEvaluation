import random
from newsroom import jsonl
from newsroom.analyze import Fragments
from newsroom.analyze.rouge import ROUGE_N
from newsroom.analyze.rouge import ROUGE_L
import pickle
import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
spacy_tokenizer = spacy.load("en_core_web_sm")

def fetch_newsroom(pickled=False):
	print("Fetching Newsroom Dataset")
	pickle_f = 'newsroom.pickle'
	if pickled:
		data = pickle.load(open(pickle_f, 'rb'))
	else:
		with jsonl.open("newsroom.jsonl") as data_file:
			raw_data = data_file.read()
		raw_data = raw_data[:100000]
		data = [{'summary' :  e['summary'], 'text': e['text'], 'coverage' : e['coverage'], 'density' : e['density'], 'compression' : e['compression']} for e in raw_data]
		pickle.dump(data, open(pickle_f, 'wb'))
	print("Fetched Newsroom Dataset with {} examples".format(len(data)))
	return data


def compute_compression(data):
	print("Computing Compression")
	return 1 - (sum([1 / ex['compression'] for ex in data]) / len(data))


def compute_ts(data):
	print("Computing Topic Similarity")
	return None


def compute_abs1(data):
	print("Computing Abstractivity-1")
	return 1 - (sum([ex['coverage'] for ex in data]) / len(data))


def compute_abs2(data):
	return None
	print("Computing Abstractivity-2")
	output = []
	for ex in tqdm(data):
		output.append(ex['density'] / len(spacy_tokenizer(ex['summary'])))
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


def compute_statistics(data):
	compression, topic_similarity, abs1, abs2, red1, red2, redL = [None] * 7
	compression = compute_compression(data)
	topic_similarity = compute_ts(data)
	abs1 = compute_abs1(data)
	abs2 = compute_abs2(data)
	red1, red2, redL = compute_red(data)
	return {"CMP" : compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL}

if __name__ == '__main__':
	newsroom_data = fetch_newsroom(pickled=True)
	statistics = compute_statistics(newsroom_data)
	print(statistics)