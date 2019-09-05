import os
from io import BytesIO
import os.path
from flask import Flask, request, render_template, url_for, redirect,make_response,send_file
import errno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import joblib
from gensim.models import Word2Vec
from coherence import convert_pdf_to_txt,split_into_sentences,preclean,topic_modeling,plot_top_term_weights
from set_parameters import SetParameters
from sklearn.decomposition import NMF
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from collections import Counter
import base64
from sklearn import preprocessing
import json

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"


app = Flask(__name__)
app.secret_key = "development-key"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/view")
def view():
    return render_template("view.html")

@app.route('/top_term_weight')
def top_term_weight():
    plots = []
    (W,H,k,dtm_terms,A,A_norm)=joblib.load("best_paras.pkl")
    for topic_index in range(k):
        plot=plot_top_term_weights( dtm_terms, H, topic_index, 8 )
        plots.append(plot)
    return render_template('result.html', articles=plots)

@app.route('/piechart')
def piechart():
    (W,H,k,dtm_terms,A,A_norm)=joblib.load("best_paras.pkl")
    X_topics_norm = preprocessing.normalize(W, norm='l2')
    columns = [str(i) for i in range(0,k)]
    dtm_to_topic = pd.DataFrame(X_topics_norm,columns=columns)
    df_group = pd.melt(dtm_to_topic,value_vars=columns)
    df_group = df_group.rename(columns={'variable':'Topics', 'value':'Score'})
    repartition = df_group.groupby(by=df_group["Topics"]).sum()
    colors_list = ['yellow', 'lightcoral', 'skyblue', 'purple', 'pink','green','blue','cyan']

    repartition['Score'].plot(kind='pie',
                            figsize=(20, 5),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct
                            colors=colors_list  # add custom colors
                             # 'explode' lowest 3 continents
                            )
    #plt.title('Topic distribution over document', y=2.0)
    plt.axis('equal')
    plt.legend(labels='Topic '+repartition.index, loc='upper right')

    fig = BytesIO()
    plt.savefig(fig,format='png')
    fig.seek(0)
    result= base64.b64encode(fig.getvalue()).decode()
    plt.close()
    #return send_file(fig,mimetype='image/png')
    return render_template('overall.html',piechart=result)


@app.route("/modeling")
def modeling():
    return render_template("modeling.html")

@app.route("/viewText", methods=['POST'])
def viewText():
    if 'document' in request.files:
        input_text = request.files['document']
        if input_text.filename != '':
            input_text.save(os.path.join('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/', input_text.filename))
            lone=convert_pdf_to_txt('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/'+ input_text.filename)
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/rawfile.txt','w',encoding='utf-8') as f:
                f.write(lone)
            f.close()
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/rawfile.txt',encoding='utf-8') as f:
                text=f.read().lower()
            f.close()
            sentences=split_into_sentences(text)
            text=' '.join(sentences)
    custom_stop_words = []
    with open( "data/stopwords.txt", "r" ) as fin:
        for line in fin.readlines():
            custom_stop_words.append( line.strip())
    img=BytesIO()
    data = WordCloud(background_color="white",stopwords=custom_stop_words).generate(text)
    plt.figure()
    plt.imshow(data,interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url=base64.b64encode(img.getvalue()).decode()
    plt.close()
    return render_template('view.html',article=plot_url)

@app.route("/preprocess", methods=['POST'])
def preprocess():
    if 'inputfile' in request.files:
        rawdata = request.files['inputfile']
        if rawdata.filename != '':

            rawdata.save(os.path.join('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/', rawdata.filename))
            lone=convert_pdf_to_txt('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/'+ rawdata.filename)
            return_url=preclean(lone)
            return send_file(return_url,attachment_filename='plot.png',mimetype='image/png')
if __name__=="__main__":
    app.run(debug=True)
