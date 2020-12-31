from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Data:
    __data = None
    @staticmethod
    def getData():
        if Data.__data == None:
            print("Here")
            Data()
        return Data.__data
    def __init__(self,counter,X_pool,y_pool,learner,committee,accuracy,X_test,y_test,classlist,queries):
        print("Inint called")
        self.counter = counter
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.learner = learner
        self.committee = committee
        self.accuracy = list(accuracy)
        self.X_test = X_test
        self.y_test = y_test
        self.classlist = classlist
        self.queries = queries
        print(type(accuracy))
        Data.__data = self

    def setdata(self,params):
        self.counter = params["counter"]
        self.X_pool = params["X_pool"]
        self.y_pool = params["y_pool"]
        print(params["accuracy"])
        print(self.accuracy)
        list = self.accuracy
        list.append(params["accuracy"])
        self.accuracy = list
        # self.accuracy = list(list(self.accuracy).append(params["accuracy"]))
        print(self.accuracy)
        Data.__data = self


    def givedata(self):
        params={}
        params["counter"] = self.counter
        params["X_pool"] = self.X_pool
        params["y_pool"] = self.y_pool
        params["learner"] = self.learner
        params["committee"] = self.committee
        params["accuracy"] = self.accuracy
        return params

