#DEISON TUIRAN LONDOÑO
#ID 014810
#ID 1003644616


import pandas as pd
import numpy as np
from sklearn import tree
import pydotplus as pyd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import statsmodels.api as sm


carseats = sm.my_datasets.get_rdataset("Carseats", "ISLR")
my_data = carseats.my_data

my_data["High_sales"] = np.where(my_data.Sales > 8, 0, 1)
my_data = my_data.drop(columns = "Sales")

shelveLocNormalized = {"Malo":0, "Medio":1, "Bueno":2}
urbanNormalized = {"Si":1, "No":0}
usNormalized = {"Si":1, "No":0}

my_data["ShelveLoc"] = my_data["ShelveLoc"].map(shelveLocNormalized)
my_data["Urban"] = my_data["Urban"].map(urbanNormalized)
my_data["US"] = my_data["US"].map(usNormalized) 

features = ["CompPrice", "Income", "Advertising", "Population", "Price", "ShelveLoc", "Age", "Education", "Urban", "US"]

x = my_data[features]
y = my_data["High_sales"]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x,y)
all_my_data = tree.export_graphviz(dtree, out_file=None, features_names=features)
graph = pyd.graph_from_dot_my_data(all_my_data)
graph.write_png("mydecisiontree.png")
img = pltimg.imread("mydecisiontree.png")
imgplot = plt.imshow(img)
plt.show()



