# @title Load the Universal Sentence Encoder's TF Hub module
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt


def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


import math


def run_sts_benchmark(sent1, sent2):
    sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(sent1)), axis=1)
    sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(sent2)), axis=1)
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    """Returns the similarity scores"""
    return scores


module_url = "/data1/zhouxukun/dynamic_backdoor_attack/universal-sentence-encoder"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print("module %s loaded" % module_url)
import numpy


def embed(input):
    return model(input)


from tqdm import tqdm
dataset_name='sst'
for dataset_name in ['sst','olid','agnews']:
    # input_sentences = open(f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/{dataset_name}/test_attacked.tsv').readlines()
    input_sentences = open(f'/data1/zhouxukun/backdoor_baseline/{dataset_name}/test_attacked.tsv').readlines()
    # input_sentences = open(
    #     f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/syntactic_attack/{dataset_name}/test_attacked.tsv').readlines()
    input_sentences=[each.strip().split('\t')[0].lower() for each in input_sentences]
    # input_sentence_2 = open(f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/{dataset_name}/test.tsv').readlines()
    input_sentence_2 = open(f'/data1/zhouxukun/backdoor_baseline/{dataset_name}/test.tsv').readlines()
    # input_sentence_2 = open(
    #     f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/syntactic_attack/{dataset_name}/test.tsv').readlines()
    input_sentence_2=[each.strip().split('\t')[0].lower() for each in input_sentence_2]
    total_cosine_similarity = []

    for idx in tqdm(range(0, len(input_sentences), 32)):
        line1 = input_sentences[idx:idx + 32]
        line2 = input_sentence_2[idx:idx + 32]
        scores = run_sts_benchmark(line1, line2)
        total_cosine_similarity.extend(scores.numpy().tolist())
    print(f"mean cosine similarity : {numpy.mean(total_cosine_similarity)}")
