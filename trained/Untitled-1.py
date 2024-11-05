# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import numpy as np
import os
import time
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# %%
DATASET_PATH='/content/drive/MyDrive/Dataset/CICIDS/preprocesd_cicids.csv'
print(DATASET_PATH)

# %%
# ,nrows=10000, header=0
start = time.time()
df=pd.read_csv(DATASET_PATH)
df.head()
print("Time taken to load the data: ", time.time()-start," seconds")

# %%
df.to_feather('output.feather')

# %%
import pyarrow.feather as feather

# Read the Feather file into a pandas DataFrame
start = time.time()
df = feather.read_feather('/content/output.feather')
print("Time taken to load the data: ", time.time()-start," seconds")
# Display the DataFrame
df.head()

# %%
df.shape

# %%
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr(numeric_only=True)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
              colname = corr_matrix.columns[i]
              col_corr.add(colname)
    return col_corr

# %%
corr_features = correlation(df, 0.85)
corr_features

# %%
df.drop(corr_features,axis=1,inplace=True)

# %%
df.shape

# %%
# create a Series with the count of rows in each group
label_counts = df[' Label'].value_counts()

# create a list of labels that have less than 10,000 rows
labels_to_merge = label_counts[label_counts < 3000].index.tolist()
print(labels_to_merge)

# %%
# create a new label called 'Other' and merge the labels with less than 10,000 rows
df[' Label'] = df[' Label'].apply(lambda x: 'Other' if x in labels_to_merge else x)
# group the rows by the new 'Label' column
grouped_df = df.groupby(' Label')

# %%
df.shape

# %%
df[' Label'].value_counts()

# %%
x = df.drop([' Label'],axis=1)
y = df[' Label']

# %%
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
rus.fit(x, y)
Xn, yn = rus.fit_resample(x, y)
# Xn.value_counts()

# %%
Xn.shape

# %%
yn.shape

# %%
yn.value_counts()

# %%
#z-score   z = (x - mean) / std
# it can make it easier for the algorithm to learn meaningful patterns in the data
cols = list(Xn.columns)
for col in cols:
    Xn[col] = stats.zscore(Xn[col])

# %%
Xn.head()

# %%
from sklearn.model_selection import  train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(Xn,yn,test_size=0.20,random_state=0)

# %%
print(np.any(np.isnan(X_train)))
print(np.all(np.isfinite(X_train)))

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Create an instance of SimpleImputer with 'mean' strategy to replace NaN values
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to X_train and transform X_train and X_test with it
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Apply StandardScaler to X_train and X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.neighbors import KNeighborsClassifier
# model training USING KNN (suppor vector machine)
start = time.time()
k = 10 # number of nearest neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)
print("Time taken to train model: ", time.time()-start," seconds")

# %%
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# performance metrics calculation of our model over training data set
start = time.time()
Predict_X =  knn.predict(X_train)
scores = cross_val_score(knn, X_train, Y_train, cv=7)
accuracy = metrics.accuracy_score(Y_train,Predict_X)
confusion_matrix = metrics.confusion_matrix(Y_train, Predict_X)
classification = metrics.classification_report(Y_train, Predict_X)
print("Time taken to for performance matric calculation: ", time.time()-start," seconds")

# %%
print()
print('--------------------------- Results --------------------------------')
print()
print ("Cross Validation Mean Score:" "\n", scores.mean())
print()
print ("Model Accuracy:" "\n", accuracy)
print()
print("Confusion matrix:" "\n", confusion_matrix)
print()
print("Classification report:" "\n", classification)
print()

# %%
def plot_confusion_matrix(cm,title,cmap=None,target=None,normalize=False):

    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target is not None:
        ticks = np.arange(len(target))
        plt.xticks(ticks, target, rotation=45)
        plt.yticks(ticks, target)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white")
    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig(title, bbox_inches='tight', dpi=300)

# %%
plot_confusion_matrix(cm=confusion_matrix ,title= 'KNN Classifire')

# %%
# performance metrics calculation of our model over test data set
start = time.time()
Predict_X =  knn.predict(X_test)
scores = cross_val_score(knn, X_test, Y_test, cv=7)
accuracy = metrics.accuracy_score(Y_test,Predict_X)
confusion_matrix = metrics.confusion_matrix(Y_test, Predict_X)
classification = metrics.classification_report(Y_test, Predict_X)
print("Time taken to for performance matric calculation: ", time.time()-start," seconds")

# %%
print()
print('--------------------------- Results --------------------------------')
print()
print ("Cross Validation Mean Score:" "\n", scores.mean())
print()
print ("Model Accuracy:" "\n", accuracy)
print()
print("Confusion matrix:" "\n", confusion_matrix)
print()
print("Classification report:" "\n", classification)
print()

# %%
plot_confusion_matrix(cm=confusion_matrix ,title= 'KNN Classifire')


