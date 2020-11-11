
from flask import Flask, render_template, request
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from IPython import display
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/next',methods=['POST'])
def query():
    n_initial = 100
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=uncertainty_sampling,
        X_training=X_initial, y_training=y_initial
    )
    data1 = request.form['queries']
    accuracy_scores = [learner.score(X_test, y_test)]

    for i in range(n_queries):
        display.clear_output(wait=True)
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
    n_queries = data1
    print(data1)


@app.route('/train', methods=['POST'])
def train():
    data1 = request.form['queries']
    print(data1)


if __name__ == '__main__':
    app.run()
