import shutil

import modAL
from flask import Flask, render_template, request
import pickle
import numpy as np
from io import StringIO
from flask import Flask, send_file
import numpy as np
import numpy
from setuptools import extension
from skimage.io import imsave
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from io import BytesIO
from PIL import Image
import os
from data import Data
import patoolib
import re

#import matplotlib



app = Flask(__name__)
app.secret_key = "super secret key"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'C:/Users/ASUS/PycharmProjects/Active Learning/static'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import zipfile
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['zip','rar'])
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,entropy_sampling
from modAL.disagreement import vote_entropy_sampling,max_disagreement_sampling,max_std_sampling,consensus_entropy_sampling

from modAL.models import ActiveLearner, Committee
from modAL.models import BayesianOptimizer
from modAL.batch import uncertainty_batch_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from functools import partial

import numpy as np

# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)
from IPython import display
# from matplotlib import pyplot as plt


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]

# @app.route('/next')
def generate_image():
    """
    Return a generated image as a png by
    saving it into a StringIO and using send_file.
    """
    num_tiles = 20
    tile_size = 30
    arr = np.random.randint(0, 255, (num_tiles, num_tiles, 3))
    arr = arr.repeat(tile_size, axis=0).repeat(tile_size, axis=1)

    # We make sure to use the PIL plugin here because not all skimage.io plugins
    # support writing to a file object.
    strIO = StringIO()
    buffer = BytesIO()
    imsave(buffer, arr, plugin='pil', format_str='png')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')




@app.route("/")
def main():
    return render_template("index.html",data=[{'name':'Random Forest'}, {'name':'KNN'}, {'name':'Decision Tree'}],
                           query=[{'name':'Uncertainty Sampling'},{'name':'Entropy Sampling'},
                                    {'name':'Random Sampling'},
                                    {'name':'Query By Committee(Uncertainty Sampling)'},
                                    {'name':'Query By Committee(Vote Entropy Sampling)'},
                                    {'name':'Query By Committee(Max Disagreement Sampling)'},
                                    {'name':'Query By Committee(Max STD Sampling)'},
                                    {'name':'Query By Committee(Consensus Entropy Sampling)'}
                                  ],
                           structure=[{'name':'Label Name given to Folder Containing Images','id':0},
                                  {'name':'Label Name given to Images','id':1}
                                  ])


@app.route('/train', methods=['POST'])
def helper():
    data = Data.getData()
    queries = data.queries
    X_test = data.X_test
    y_test = data.y_test
    X_pool = data.X_pool
    y_pool = data.y_pool
    counter = data.counter
    learner = data.learner
    committee = data.committee
    accuracy = data.accuracy
    classlist = data.classlist
    print(classlist)
    print(counter)
    print(queries)
    if(int(counter)==int(queries)):
        print("First Condition")
        print(classlist)
        if(learner != None):
            query_idx, query_inst = learner.query(X_pool)
            print("Learner")
            print(learner)
        elif(committee!=None):
            query_idx, query_inst = committee.query(X_pool)
        try:
            arr = query_inst.reshape(200,200,3)
        except:
            arr = query_inst.reshape(200,200)
        print(arr)
        rescaled = (255.0 / arr.max() * (arr - arr.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        new_size = (300, 300)
        im = im.resize(new_size)
        filename = secure_filename("image.png")
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(filename)))
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        params = {}
        params["X_pool"] = X_pool
        params["y_pool"] = y_pool
        params["counter"] = int(counter)-1

        if learner!=None:
            params["accuracy"] = learner.score(X_test,y_test)
        elif committee!=None:
            params["accuracy"] = committee.score(X_test, y_test)
        data.setdata(params)
        return render_template("after.html",classlist=classlist,query=[{'name':'Uncertainty Sampling'},{'name':'Entropy Sampling'},
                                                                       {'name':'Random Sampling'},
                                                                       {'name':'Query By Committee(Uncertainty Sampling)'},
                                                                       {'name':'Query By Committee(Vote Entropy Sampling)'},
                                                                       {'name':'Query By Committee(Max Disagreement Sampling)'},
                                                                       {'name':'Query By Committee(Max STD Sampling)'},
                                                                       {'name':'Query By Committee(Consensus Entropy Sampling)'}

                                                                       ])
    if(int(counter)>=1):
        if(learner != None):
            query_idx, query_inst = learner.query(X_pool)
            print("Learner")
            print(learner)
        elif(committee!=None):
            query_idx, query_inst = committee.query(X_pool)
        print(query_inst.shape)
        print(query_inst)
        try:
            arr = query_inst.reshape(200,200,3)
        except:
            arr = query_inst.reshape(200,200)
        print(arr)
        rescaled = (255.0 / arr.max() * (arr - arr.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        new_size = (300, 300)
        im = im.resize(new_size)
        filename = secure_filename("image.png")
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(filename)))
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
        y_new = np.array([int(request.form['queries'])],dtype=int)
        if(learner!=None):
            learner.teach(query_inst.reshape(1, -1), y_new)
        elif(committee!=None):
            committee.teach(query_inst.reshape(1, -1), y_new)
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        params = {}
        params["X_pool"] = X_pool
        params["y_pool"] = y_pool
        params["counter"] = int(counter)-1
        if learner!=None:
            params["accuracy"] = learner.score(X_test,y_test)
        elif committee!=None:
            params["accuracy"] = committee.score(X_test, y_test)
        data.setdata(params)
        accuracy_string = ""
        count = 0
        iterations = ""
        for i in data.accuracy:
            n = float(i)
            n*=100
            accuracy_string +=str(n)
            accuracy_string +=","
            iterations+=str(count)
            iterations+=","
            count+=1
        accuracy_string = accuracy_string[:-1]
        iterations = iterations[:-1]
        print("Accuracy string",accuracy_string)
        return render_template("after.html",data = accuracy_string,iteration = iterations,classlist=classlist)
    else:
        accuracy_string = ""
        iterations = ""
        count = 0
        for i in data.accuracy:
            n = float(i)
            n *= 100
            accuracy_string += str(n)
            accuracy_string += ","
            iterations += str(count)
            iterations += ","
            count += 1
        accuracy_string = accuracy_string[:-1]
        iterations = iterations[:-1]
        print("Final",accuracy_string,iterations)
        return render_template("final.html",accuracy = float(data.accuracy[-1])*100,data = accuracy_string,iteration = iterations,)


@app.route('/next',methods=['GET','POST'])
def query():
    # n_initial = 100
    # X, y = load_digits(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    #
    # initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    #
    # X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    # X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
    strategy = None
    classifier = None

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    filename = secure_filename(file.filename)
    # shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'],filename.split(".")[0]))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        if(filename.split(".")[1]=="rar"):
            patoolib.extract_archive(os.path.join(UPLOAD_FOLDER, filename), outdir=os.path.join(UPLOAD_FOLDER))
        else:
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()
            print("Succesfull")

    st = request.form.get('strategy_select')
    cl = request.form.get('classifier_select')
    option = int(request.form.get('structure_select'))
    print(cl)
    if(str(cl)=='Random Forest'):
        classifier = RandomForestClassifier()
    elif(str(cl)=='KNN'):
        classifier = KNeighborsClassifier()
    else:
        classifier = DecisionTreeClassifier()

    n_queries = request.form['queries']

    print(st)
    classlist =[]
    classes = {}
    data = {}
    data['image'] = []
    data['label'] = []
    filename = secure_filename(file.filename)
    print(filename)
    if option == 0:
        for dirname, _, filenames in os.walk(os.path.join(UPLOAD_FOLDER,filename.split(".")[0])):
            for filename in filenames:
                if('.jpg' in filename or 'jpeg' in filename):
                    image = Image.open(os.path.join(dirname, filename))
                    image = image.resize((200,200), Image.ANTIALIAS)
                    size = np.array(image).size
                    if(len(classes)==0):
                        data['image'] = np.array(numpy.array(image)).reshape((1,size))
                    else:
                        try:
                            x = numpy.array(image).reshape((1,size))
                            data['image'] = np.append(data['image'],x,axis=0)
                        except:
                            continue
                    if(dirname.split('\\')[-1] not in classes.keys()):
                        classlist.append({'name':dirname.split('\\')[-1],'number':len(classes)})
                        classes[dirname.split('\\')[-1]] = len(classes)
                            #print(os.path.join(dirname, filename))
                            #print(dirname)

                    data['label'].append(classes[dirname.split('\\')[-1]])
                    print(classes)
    else:
        for imfile in os.listdir(os.path.join(UPLOAD_FOLDER,filename.split(".")[0])):
            if imfile.endswith(".jpg") or imfile.endswith(".jpeg"):

                image = Image.open(os.path.join(os.path.join(UPLOAD_FOLDER,filename.split(".")[0]), imfile))
                image = image.resize((200,200), Image.ANTIALIAS)
                size = np.array(image).size

                if(len(classes)==0):
                    data['image'] = np.array(numpy.array(image)).reshape((1,size))
                else:
                    try:
                        x = numpy.array(image).reshape((1,size))
                        data['image'] = np.append(data['image'],x,axis=0)
                    except:
                        continue
                if(("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0]))) not in classes.keys()):
                    classlist.append({'name':("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0]))),'number':len(classes)})
                    classes[("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0])))] = len(classes)
                data['label'].append(classes[("".join(re.split("[^a-zA-Z]*",imfile.split(".")[0])))])
                print(classes)
            else:
                continue

    X = data['image']
    y = data['label']
    n_initial = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial=[]
    y_initial = []
    print(type(X_initial))
    for i in range(n_initial):

        v = np.array(X_train[initial_idx[i]]).reshape((1,size))

        #print(v.shape)
        y_initial.append(y_train[i])
        if(i==0):
            X_initial = np.array(X_train[initial_idx[i]]).reshape((1,size))

            print(X_initial.shape)
        else:
            X_initial = np.append(X_initial,v,axis=0)
        #print("X Shape",X_initial.shape)
        #     X_initial = X_initial.append(X_train[initial_idx[i]])
    X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
    print(X.shape)
    print(X[0].shape)
    print(X_initial.shape)

    params = {}
    params["X_test"] = X_test
    params["y_test"] = y_test
    params["counter"] = n_queries
    params["X_pool"] = X_pool
    params["y_pool"] = y_pool
    if(str(st)=='Uncertainty Sampling'):

        print(classifier)
        print(cl)
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=uncertainty_sampling,
            X_training=X_initial, y_training=y_initial
        )

        params["learner"] = learner
        accuracy_scores = learner.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,learner,None,accuracy,X_test,y_test,classlist,n_queries)
        helper()
    elif(str(st)=='Entropy Sampling'):

        print(classifier)
        print(cl)
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=entropy_sampling,
            X_training=X_initial, y_training=y_initial
        )

        params["learner"] = learner
        accuracy_scores = learner.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,learner,None,accuracy,X_test,y_test,classlist,n_queries)
        helper()
    elif(str(st)=='Random Sampling'):
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=random_sampling,
            X_training=X_train, y_training=y_train
        )
        accuracy_scores = learner.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,learner,None,accuracy,X_test,y_test,classlist,n_queries)
        helper()
    elif(str(st)=='Query By Committee(Vote Entropy Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=vote_entropy_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries)
        helper()

    elif(str(st)=='Query By Committee(Uncertainty Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=uncertainty_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries)
        helper()

    elif(str(st)=='Query By Committee(Max Disagreement Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=max_disagreement_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries)
        helper()

    elif(str(st)=='Query By Committee(Max STD Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=max_std_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries)
        helper()

    elif(str(st)=='Query By Committee(Consensus Entropy Sampling)'):
        learner1 = ActiveLearner(
            estimator = RandomForestClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner2 = ActiveLearner(
            estimator=KNeighborsClassifier(),
            X_training=X_train,y_training=y_train
        )
        learner3 = ActiveLearner(
            estimator=DecisionTreeClassifier(),
            X_training=X_train,y_training=y_train
        )
        committee = Committee(
            learner_list=[learner1,learner2,learner3],
            query_strategy=consensus_entropy_sampling
        )
        params["committee"] = committee
        accuracy_scores = committee.score(X_test, y_test)
        params["accuracy"] = accuracy_scores
        print(accuracy_scores)
        accuracy = []
        accuracy.append(accuracy_scores)
        data = Data(n_queries,X_pool,y_pool,None,committee,accuracy,X_test,y_test,classlist,n_queries)
        helper()

    return render_template("after.html",data=n_queries,accuracy = accuracy_scores)

app.run(debug=True)
