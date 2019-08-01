import os
import io
import os.path
from flask import Flask, request, render_template, url_for, redirect,make_response,send_file
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
import joblib
from gensim.models import Word2Vec
from coherence import coherence_output,get_docs_per_topic,get_descriptor,get_top_snippets,get_words_per_topic_df
from set_parameters import SetParameters
import collections
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import base64

image_size = {'1024×576':"10.24,5.76", "1280×720(HD)":"12.80,7.20", "1366×768":"13.66,7.68",
"1920×1080(FHD)":"19.20,10.80"}

app = Flask(__name__)
UPLOAD_FOLDER = './media'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route("/viewText", methods=['POST'])
def viewText():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            photo.save(os.path.join('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/', photo.filename))
            lone=convert_pdf_to_txt('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/'+ photo.filename)
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/rawfile.txt','w',encoding='utf-8') as f:
                f.write(lone)

            raw_documents=[]
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/rawfile.txt',encoding='utf-8') as f:
                for line in f.readlines():
                    text=line.strip().lower()
                    raw_documents.append(text)
            text=' '.join(raw_documents)

    stopwords = STOPWORDS
    stopwords.add('will')
    img=io.BytesIO()

    data = WordCloud(background_color="white",stopwords=stopwords).generate(text)
    plt.figure()
    plt.imshow(data,interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url=base64.b64encode(img.getvalue()).decode()

    return render_template('view.html',output_image=plot_url)

@app.route("/modeling",methods=["GET","POST"])
def modeling():
    paras=SetParameters()
    if request.method=="POST":
        if paras.validate()==False:
            return render_template("modeling.html",form=paras)
        else:
            k_topic=paras.no_of_topic.data
            k_words=paras.no_of_words.data
            k_document=paras.no_of_documents.data
            (A,terms,dubby)=joblib.load("articles-tfidf.pkl")

            model=decomposition.NMF(init="nndsvd",random_state=42,alpha=0.1,l1_ratio=0.5,n_components=k_topic)
            W=model.fit_transform(A)
            H=model.components_

            wordpertopicresult=get_words_per_topic_df(H,terms,k_words)
            documentpertopicresult=get_docs_per_topic(W,dubby,k_document)
            wordpertopicresult.index=range(1,k_words+1)
            documentpertopicresult.index=range(1,k_document+1)
            return render_template('result.html',word_per_topic=wordpertopicresult.to_html(),docs_per_topic=documentpertopicresult.to_html())

    elif request.method=="GET":
        return render_template("modeling.html",form=paras)



@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':

            photo.save(os.path.join('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/', photo.filename))
            lone=convert_pdf_to_txt('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/'+ photo.filename)

            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/test.txt','w',encoding='utf-8') as f:
                f.write(lone)

            raw_documents=[]
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/test.txt',encoding='utf-8') as f:

                for line in f.readlines():
                    text=line.strip().lower()
                    raw_documents.append(text)
            shear=[i.replace('\xe2\x80\x9c','') for i in raw_documents ]
            shear=[i.replace('\xe2\x80\x9d','') for i in shear ]
            shear=[i.replace('\xe2\x80\x99s','') for i in shear ]
            shears = [x for x in shear if x != ' ']
            shearss = [x for x in shears if x != '']
            dubby=[re.sub("[^a-zA-Z]+", " ", s) for s in shearss]
            custom_stop_words = []
            with open( "data/stopwords.txt", "r" ) as fin:
                for line in fin.readlines():
                    custom_stop_words.append(line.strip())
            vectorizer = TfidfVectorizer(analyzer='word', max_df=0.95,stop_words=custom_stop_words, min_df = 2)
            A = vectorizer.fit_transform(dubby)
            terms = vectorizer.get_feature_names()
            joblib.dump((A,terms,dubby),"articles-tfidf.pkl")


            kmin, kmax = 4, 20
            topic_models = []
            for k in range(kmin,kmax+1):
                model = decomposition.NMF( init="nndsvd",n_components=k )
                W = model.fit_transform( A )
                H = model.components_

                topic_models.append( (k,W,H) )
            sentences=[nltk.word_tokenize(sentence) for sentence in dubby]
            model=Word2Vec(sentences,size=500,min_count=1,sg=1)

            url=coherence_output(topic_models,model,terms)

            return send_file(url,attachment_filename='plot.png',mimetype='image/png')
            return 'success'+photo.filename

if __name__=="__main__":
    app.run(debug=True)
