import os
import errno
from io import BytesIO
import os.path
from flask import Flask,flash,request, redirect, url_for, render_template, send_from_directory,redirect,make_response,send_file
from werkzeug.utils import secure_filename
import errno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import joblib
from gensim.models import Word2Vec
from coherence import display_labels,label_document,convert_pdf_to_txt,label_document,split_into_sentences,preclean,topic_modeling,plot_top_term_weights
from sklearn.decomposition import NMF
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from wordcloud import WordCloud
import seaborn as sns
from time import time
sns.set_style('whitegrid')

#from nltk.tokenize import word_tokenize
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

UPLOAD_FOLDER = 'F:/uploads/'
#TEST_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/test/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf'])
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/view")
def view():
    return render_template("view.html")

@app.route("/rawView")
def rawView():
    (raw,clean_sentences)=joblib.load("raw_clean.pkl")

    return render_template("contents.html",contents=raw)

@app.route("/afterClean")
def afterClean():
    (raw,clean_sentences)=joblib.load("raw_clean.pkl")


    return render_template("contents.html",contents=clean_sentences)

@app.route('/viewText', methods=['POST'])
def viewText():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('view.html')
        file = request.files['file']
        if file.filename == '':
            flash('No file selected to visualize')
            return render_template('view.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            folder_path = 'input_pdf/'
            _mkdir_p(folder_path)
            file.save(os.path.join(folder_path, filename))
            lone=convert_pdf_to_txt(folder_path+ filename)

            with open(folder_path+'rawfile.txt','w',encoding='utf-8') as f:
                f.write(lone)
            f.close()

            raw_documents=[]
            with open(folder_path+'rawfile.txt','r',encoding='utf-8') as f:
                for line in f.readlines():
                    text=line.strip().lower()
                    raw_documents.append(text)
            text=' '.join(raw_documents)

            custom_stop_words = []
            with open( "data/stopwords.txt", "r" ) as fin:
                for line in fin.readlines():
                    custom_stop_words.append( line.strip() )
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
            #flash('File successfully uploaded')
            return render_template('view.html',article=plot_url)

        else:
            flash('Please enter PDF file only')
            return render_template('view.html')

@app.route('/top_term_weight')
def top_term_weight():
    plots = []
    (W,H,k,dtm_terms,A,tfidf,clean_data)=joblib.load("best_paras.pkl")
    array=label_document(clean_data)
    df=array.to_frame(name='Name')
    my_list = df["Name"].tolist()
    #data=(my_list,len(my_list))

    for topic_index in range(k):
        plot=plot_top_term_weights( dtm_terms, H, topic_index, 8)
        plots.append(plot)
    return render_template('result.html', articles=plots,result=my_list)

@app.route('/piechart')
def piechart():
    (W,H,k,dtm_terms,A,tfidf,clean_data)=joblib.load("best_paras.pkl")
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
    terms=tfidf.get_feature_names()

    total_counts = np.zeros(len(terms))
    for t in A:
        total_counts+=t.toarray()[0]
    count_dict = (zip(terms, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:8]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.figure(2, figsize=(10, 8/1.1180))
    plt.subplot(title='8 most common words in document')
    sns.set_context("notebook", font_scale=1.15, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')

    bytes_image = BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    graph= base64.b64encode(bytes_image.getvalue()).decode()
    plt.close()
    return render_template('overall.html',piechart=result,most_common=graph)


@app.route("/modeling")
def modeling():
    return render_template("modeling.html")

@app.route("/preprocess", methods=['POST'])
def preprocess():
    if 'file' in request.files:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return render_template('modeling.html')
            file = request.files['file']
            if file.filename == '':
                flash('No file selected to process')
                return render_template('modeling.html')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                folder_path = 'input_pdf/'
                _mkdir_p(folder_path)

                file.save(os.path.join(folder_path, filename))
                lone=convert_pdf_to_txt(folder_path+ filename)

                return_url,clean_sentences=preclean(lone)

                joblib.dump((lone,clean_sentences),"raw_clean.pkl")
                return send_file(return_url,attachment_filename='plot.png',mimetype='image/png')
                
if __name__=="__main__":
    app.run(debug=True)
