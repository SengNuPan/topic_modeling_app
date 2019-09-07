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
from coherence import convert_pdf_to_txt,split_into_sentences,preclean,topic_modeling,plot_top_term_weights
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

@app.route('/viewText', methods=['POST'])
def viewText():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            folder_path = 'input_pdf/'
            _mkdir_p(folder_path)
            file.save(os.path.join(folder_path, filename))
            lone=convert_pdf_to_txt(folder_path+ filename)
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/rawfile.txt','w',encoding='utf-8') as f:
                f.write(lone)
            f.close()

            raw_documents=[]
            with open('C:/Users/Seng Nu Pan/Desktop/topic_modeling_wiht_flask/data/rawfile.txt','r',encoding='utf-8') as f:
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
            flash('Allowed file types are txt, pdf')
            return render_template('view.html')

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

@app.route("/preprocess", methods=['GET','POST'])
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
