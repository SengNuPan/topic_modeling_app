import numpy as np
from io import BytesIO
import re
import pandas as pd
from time import time
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
sns.set_style('whitegrid')
import base64
matplotlib.use('Agg')
from itertools import combinations
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

import os
import joblib

from flask import Flask, request, render_template, url_for, redirect,send_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn import decomposition
from sklearn import preprocessing

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

import gensim
import errno
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

#nltk.data.path.append('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/nltk_data')

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"

custom_stop_words = []
with open( "data/stopwords.txt", "r" ) as fin:
    for line in fin.readlines():
        custom_stop_words.append(line.strip())
    fin.close()

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text

def calculate_coherence( model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            pair_scores.append( model.similarity(pair[0], pair[1]) )
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms

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
    bytes_image = BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plt.close(fig)
    #return base64.b64encode(bytes_image.getvalue()).decode()
    return bytes_image,best_k


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub("\'","",text)
    text=re.sub('\s+', ' ', text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    text = text.replace("e.g.","e<prd>g<prd>")
    text=re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", text)
    text=re.sub("\s*-\s*", "", text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    lemmatizer=WordNetLemmatizer()
    for i in range(len(sentences)):
        words=nltk.word_tokenize(sentences[i])
        newwords=[lemmatizer.lemmatize(word)for word in words]
        sentences[i]=' '.join(newwords)
    return sentences

class TokenGenerator:
    def __init__( self, documents, stopwords ):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )
    def __iter__( self ):
        #print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall( doc ):
                if tok in self.stopwords:
                    tokens.append( "<stopword>" )
                elif len(tok) >= 2:
                    tokens.append( tok )
            yield tokens

def topic_modeling(n_components, init="nndsvd",alpha=.1, l1_ratio=.5):
    nmf_f = NMF(n_components=n_components, random_state=10,alpha=alpha,l1_ratio=l1_ratio)
    return nmf_f

def display_labels(H, W,documents):
    result=[]
    for topic_idx, topic in enumerate(H):
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][:3]

        for doc_index in top_doc_indices:
            result.append(documents[doc_index])
    return result

def plot_top_term_weights( terms, H, topic_index, top):

    top_indices = np.argsort( H[topic_index,:] )[::-1]
    top_terms = []
    top_weights = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
        top_weights.append( H[topic_index,term_index] )
    top_terms.reverse()
    top_weights.reverse()
    fig = plt.figure(figsize=(7,4))
    img=BytesIO()
    # add the horizontal bar chart
    ypos = np.arange(top)
    ax = plt.barh(ypos, top_weights, align="center", color="#20B2AA",tick_label=top_terms)
    (W,H,k,dtm_terms,A,tfidf,clean_data)=joblib.load("best_paras.pkl")
    result=label_document(display_labels(H,W,clean_data))
    df=result.to_frame(name='Name')
    my_list = df["Name"].tolist()
    #for i in range(len(my_list)):
    plt.title(my_list[0],fontsize=10)

    plt.xlabel("Term Weight",fontsize=12)
    plt.tight_layout()
    plt.savefig(img,format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return img_b64

def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def label_document(sentence):
    count_vect = CountVectorizer(stop_words= custom_stop_words, min_df =5,max_df=.92)
    X_train_counts = count_vect.fit_transform(newsgroups_train.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, newsgroups_train.target)
    X_new_counts = count_vect.transform(sentence)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(sentence, predicted):
        document_category = dict(zip(sentence, predicted))
    doc_cat = [ [k,v] for k, v in document_category.items()]

    columns = [str(i) for i in range(0,2)]
    df_doc_cat = pd.DataFrame(doc_cat,columns=columns)
    df_group = df_doc_cat.rename(columns={'0':'Text', '1':'Label'})

    after_gp=df_group.groupby('Label', as_index=False).count()
    data = {'Label': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],'Name': ['Athesim','Graphics','MS-Windows','IBM,PC-Hardware','MAC,Hardware','Windows.X','MISC,Forsale','Autos','Motorcycles','Sport,Baseball','Sport,Hockey','Science,Crypt','Science,Electronic','Science,Med','Science,Space','Society,Religion-Christian','Politics,Guns','Politics,Mideast','Politics,Misc','Religion,Misc']}
    label_name=pd.DataFrame(data)
    df_group = df_doc_cat.rename(columns={'0':'Text', '1':'Label'})
    df = pd.merge(after_gp, label_name, on='Label', how='left')
    a=[]
    df=df[df['Text']==df['Text'].max()]
    a.append(df['Name'])
    return a[0]

def preclean(rawtext):
    folder_path = 'input_pdf/'
    _mkdir_p(folder_path)
    with open(folder_path+'test.txt','w',encoding='utf-8') as f:
        f.write(rawtext)
    f.close()
    with open(folder_path+'test.txt','r',encoding='utf-8') as f:
        data=f.read().lower()
    f.close()
    sentences=split_into_sentences(data)

    vectorizer = TfidfVectorizer(stop_words= custom_stop_words, min_df =5,max_df=.92) #custom_stop_words
    A = vectorizer.fit_transform(sentences)
    A_norm = preprocessing.normalize(A,norm='l2')
    terms=vectorizer.get_feature_names()
    kmin, kmax = 2,8
    topic_models = []
    for k in range(kmin,kmax+1):
        model = NMF( init="nndsvd", n_components=k,random_state=10,alpha=0.1,l1_ratio=0.5)
        W = model.fit_transform( A )
        H = model.components_
        topic_models.append( (k,W,H) )
    docgen = TokenGenerator(sentences, custom_stop_words )
    w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=2, sg=1)
    url,k=coherence_output(topic_models,w2v_model,terms)
    W = topic_models[k-kmin][1]
    H = topic_models[k-kmin][2]
    joblib.dump((W,H,k,terms,A,vectorizer,sentences),"best_paras.pkl")
    return url,sentences
    #return send_file(url,attachment_filename='plot.png',mimetype='image/png')
