
from flask import Flask, render_template, request
import pickle
import numpy as np
from io import StringIO
from flask import Flask, send_file
import numpy as np
from setuptools import extension
from skimage.io import imsave
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from io import BytesIO
from PIL import Image
import os
from data import Data



#import matplotlib



app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'C:/Users/ASUS/PycharmProjects/Active Learning/static'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from IPython import display
# from matplotlib import pyplot as plt



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
    return render_template("index.html",data=[{'name':'Random Forest'}, {'name':'KNN'}, {'name':'Decision Tree'}])





@app.route('/train',methods=['POST'])
def helper():
    data = Data.getData()
    X_test = data.X_test
    y_test = data.y_test
    X_pool = data.X_pool
    y_pool = data.y_pool
    counter = data.counter
    learner = data.learner
    accuracy = data.accuracy
    print(counter)
    print(accuracy)
    if(int(counter)>=1):
        query_idx, query_inst = learner.query(X_pool)
        arr = query_inst.reshape(8, 8)
        print(arr)
        rescaled = (255.0 / arr.max() * (arr - arr.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        new_size = (300, 300)
        im = im.resize(new_size)
        filename = secure_filename("image.png")
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(filename)))
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
        y_new = np.array([int(request.form['queries'])],dtype=int)
        learner.teach(query_inst.reshape(1, -1), y_new)
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        params = {}
        params["X_pool"] = X_pool
        params["y_pool"] = y_pool
        params["counter"] = int(counter)-1
        params["accuracy"] = learner.score(X_test,y_test)
        data.setdata(params)
        accuracy_string = ""
        count = 1
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
        return render_template("after.html",data = accuracy_string,iteration = iterations)
    else:
        return render_template("final.html",accuracy = data.accuracy[-1])


@app.route('/next',methods=['POST'])
def query():
    n_initial = 100
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
    classifier = None
    cl = request.form.get('classifier_select')
    if(str(cl)=='Random Forest'):
        classifier = RandomForestClassifier()
    elif(str(cl)=='KNN'):
        classifier = KNeighborsClassifier()
    else:
        classifier = DecisionTreeClassifier()
    print(classifier)
    print(cl)
    learner = ActiveLearner(
        estimator=classifier,
        query_strategy=uncertainty_sampling,
        X_training=X_initial, y_training=y_initial
    )
    n_queries = request.form['queries']
    params = {}
    params["X_test"] = X_test
    params["y_test"] = y_test
    params["counter"] = n_queries
    params["X_pool"] = X_pool
    params["y_pool"] = y_pool
    params["learner"] = learner

    accuracy_scores = learner.score(X_test, y_test)
    params["accuracy"] = accuracy_scores

    data = Data(n_queries,X_pool,y_pool,learner,0,X_test,y_test)
    helper()
    # query_idx, query_inst = learner.query(X_pool)
    # data = query_inst.reshape(8,8)
    # rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    # im = Image.fromarray(rescaled)
    # new_size = (300,300)
    # im = im.resize(new_size)
    # filename = secure_filename("image.png")
    # im.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))

    # generate_image(query_inst.reshape(8, 8))
    # try:
    #     image = request.files['image']
    #     nom_image = secure_filename(image.filename)
    #     image = Image.open(image)
    #     ...
    #     img_io = BytesIO()
    #     image.save(img_io, extension.upper(), quality=70)
    #     img_io.seek(0)
    #     return send_file(img_io, mimetype='image/jpeg', attachment_filename=nom_image, as_attachment=True)
    # except Exception as e:
    #     print(e)
    return render_template("after.html",data=n_queries,accuracy = accuracy_scores)
    for i in range(n_queries):
        query_idx, query_inst = learner.query(X_pool)
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Digit to label')
            plt.imshow(query_inst.reshape(8, 8))
            plt.subplot(1, 2, 2)
            plt.title('Accuracy of your model')
            plt.plot(range(i + 1), accuracy_scores)
            plt.scatter(range(i + 1), accuracy_scores)
            plt.xlabel('number of queries')
            plt.ylabel('accuracy')
            display.display(plt.gcf())
            plt.close('all')

        print("Which digit is this?")
        y_new = np.array([int(input())], dtype=int)
        learner.teach(query_inst.reshape(1, -1), y_new)
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        accuracy_scores.append(learner.score(X_test, y_test))
    for i in range(n_queries):
        query_idx, query_inst = learner.query(X_pool)
        data = query_inst.reshape(8, 8)
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        new_size = (300, 300)
        im = im.resize(new_size)
        filename = secure_filename("image.png")
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
        y_new = np.array([int(input())], dtype=int)
        learner.teach(query_inst.reshape(1, -1), y_new)
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
        accuracy_scores.append(learner.score(X_test, y_test))
    n_queries = data1
    print(data1)

app.run(debug=True)