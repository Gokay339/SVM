!pip install scikit-learn==0.23.1

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
%matplotlib inline
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()

#**Veri Özellikleri ve Sınıflandırma**

# ID Alanı: Hasta tanımlayıcılarını içerir.

#Özellikler (Clump'dan Mit'e kadar): Her hastadan alınan hücre örneklerinin özelliklerini içerir. Değerler 1'den 10'a kadar puanlanmış olup, 1 benign'e en yakın değeri temsil eder.

#Class Alanı: Tanının doğruluğunu gösterir; benign (değer = 2) veya malignant (değer = 4) olarak sınıflandırılmıştır.

#**Clump kalınlığı ve Hücre boyutunun Üniformluğu'na göre sınıfların dağılımını inceleyelim:**

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

cell_df.dtypes

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

# Modelin Class değerini (yani, iyi huylu (=2) veya kötü huylu (=4)) tahmin etmesini istiyoruz.

y = np.asarray(cell_df['Class'])
y [0:5]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Eğitim Verisi:', X_train.shape,  y_train.shape)
print ('Test Verisi:', X_test.shape,  y_test.shape)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

#Model yerleştirildikten sonra yeni değerleri tahmin etmek için kullanılabilir:

yhat = clf.predict(X_test)
yhat [0:5]

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


#**Scikit-learn kütüphanesinden f1_score fonksiyonunu da kolayca kullanabilirsiniz.**

from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')

#**Doğruluk için Jaccard indeksini deneyelim:**

from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=2)
