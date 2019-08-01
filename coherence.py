import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})
import os
from flask import Flask, request, render_template, url_for, redirect
import re
import pandas as pd
from time import time
import nltk
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from gensim.models import Word2Vec

def calculate_coherence( model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):

            pair_scores.append( model.similarity(pair[0], pair[1]) )

                # If word is not in the word2vec model then as score 0.5

        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

def get_words_per_topic_df(H, feature_names, num_top_words):
    '''Returns the num_top_words words per topic in a dataframe
       Args:
           H: matrix returned by the NMF model, KxV (V = vocabulary size, #words in the corpus)
           feature_names: the word names, to map it from the indices
           num_top_words: number of words per topic
       Returns: a dataframe
    '''
    word_dict = {};
    for topic_idx, topic_vec in enumerate(H):
        words = get_words_per_topic(topic_vec, feature_names, num_top_words)
        word_dict["Topic %d:" % (topic_idx + 1)] = words
#         word_dict[(topic_idx + 1)] = words
    return pd.DataFrame(word_dict)

def get_docs_per_topic(W, documents, num_top_docs):
    '''Returns the num_top_docs documents with highest score per topic
       Args:
           W: matrix returned by the NMF model, KxD (D = #docs in the corpus)
           documents: list of documents, to map it from the indices
           num_top_docs: number of documents to show per topic
       Returns: a dataframe'''
    doc_dict = {}
    for topic_idx in range(np.shape(W)[1]):
        top_doc_indices = np.argsort(W[:,topic_idx])[::-1][0:num_top_docs]
        docs = [documents[doc_index] for doc_index in top_doc_indices]
        doc_dict["Topic %d:" % (topic_idx + 1)] = docs
    return pd.DataFrame(doc_dict)

def get_docs_per_topic_with_info(W, documents, num_top_docs):
    ''' Returns the num_top_docs document docuements and all the info from the original dataframe per topic.
        Args:
           W: matrix returned by the NMF model, KxD (D = #docs in the corpus)
           documents: original dataframe with all the data
           num_top_docs: number of documents to show per topic
        Returns: a dictionary of dataframes
    '''
    doc_dict = {};
    for topic_idx in range(np.shape(W)[1]):
        top_doc_indices = np.argsort(W[:,topic_idx])[::-1][0:num_top_docs]
        docs = pd.DataFrame(documents.iloc[top_doc_indices])
        doc_dict[topic_idx] = docs
    return doc_dict

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms

def get_top_snippets(all_snippets,W,topic_index,top):
    top_indices=np.argsort(W[:,topic_index])[::-1]
    top_snippets=[]
    for doc_index in top_indices[0:top]:
        top_snippets.append(all_snippets[doc_index])
    return top_snippets

def coherence_output(trained_models,word2vecmodel,topTerms):
    k_values = []
    coherences = []
    for (k,W,H) in trained_models:
        # Get all of the topic descriptors - the term_rankings, based on top 10 terms
        term_rankings = []
        for topic_index in range(k):
            term_rankings.append( get_descriptor( topTerms, H, topic_index, 10 ) )
        # Now calculate the coherence based on our Word2vec model
        k_values.append( k )
        coherences.append( calculate_coherence( word2vecmodel, term_rankings ) )


    fig = plt.figure(figsize=(13,7))
    # create the line plot
    ax = plt.plot( k_values, coherences )
    plt.xticks(k_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Mean Coherence")
    # add the points
    plt.scatter( k_values, coherences, s=120)
    # find and annotate the maximum point on the plot
    ymax = max(coherences)
    xpos = coherences.index(ymax)
    best_k = k_values[xpos]
    plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
    # show the plot
    plt.show()
# here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    #return base64.b64encode(bytes_image.getvalue()).decode()
    return bytes_image

def get_words_per_topic(topic_vec, feature_names, num_top_words):
    '''
    Returns a list with the num_top_words with the highest score for the topic given
    '''
    return [feature_names[i] for i in topic_vec.argsort()[:-num_top_words - 1:-1]]

def get_words_per_topic_df(H, feature_names, num_top_words):
    '''Returns the num_top_words words per topic in a dataframe
       Args:
           H: matrix returned by the NMF model, KxV (V = vocabulary size, #words in the corpus)
           feature_names: the word names, to map it from the indices
           num_top_words: number of words per topic
       Returns: a dataframe
    '''
    word_dict = {};
    for topic_idx, topic_vec in enumerate(H):
        words = get_words_per_topic(topic_vec, feature_names, num_top_words)
        word_dict["Topic %d:" % (topic_idx + 1)] = words
#         word_dict[(topic_idx + 1)] = words
    return pd.DataFrame(word_dict)
