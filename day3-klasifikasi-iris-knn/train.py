# --> Import Libraries <--

# import library Pandas untuk menggunakan kelas DataFrame untuk menampung dataset
from pandas import read_csv

# import library NumPy untuk melakukan perhitungan matematis
import numpy as np

# import library matplotlib dan seaborn untuk membuat plot
from matplotlib import pyplot
from seaborn import pairplot

# import algoritma ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# import picke untuk menyimpan model
import pickle

# -----------------------------------------------------------------------------------
# 1)                              DATA INGESTION
# -----------------------------------------------------------------------------------

# definisikan lokasi file yang akan di ambil
file_path = r'dataset\\iris.data'

# definisikan kolom yang akan di load
attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# baca data ke dalam DataFrame dari library Pandas
dataset = read_csv(file_path, names=attributes)

# -----------------------------------------------------------------------------------
# 2)                              DATA EXPLORATION
# -----------------------------------------------------------------------------------

# melihat 5 data teratas
print(dataset.head())

# melihat banyaknya data berdasarkan spesies
print(dataset.groupby('class').size())

# melihat hubungan antara semua grup
pairplot(dataset, hue='class', size=3)
pyplot.show()

# -----------------------------------------------------------------------------------
# 3)                                PREPROCESSING
# -----------------------------------------------------------------------------------

# memisahkan data menjadi feature dan label
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# membagi data untuk training dan evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# melakukan penskalaan data agar semua data memiliki skala yang sama
# dengan StandardScaler, kita membuat agar data yang kita punya
# memiliki mean = 0 dan standar deviasi = 1
scaler = StandardScaler()

# kita training scaler yang kita punya dengan data
scaler.fit(x_train)

# lakukan transformasi pada data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# -----------------------------------------------------------------------------------
# 4)                                 TRAINING
# -----------------------------------------------------------------------------------

# membuat objek classifier dari kelas KNeighborsClassifier
# dengan k = 5
classifier = KNeighborsClassifier(n_neighbors=3)

# melakukan training KNN dengan dataset dan labelnya
classifier.fit(x_train, y_train)

# -----------------------------------------------------------------------------------
# 5)                                 EVALUATION
# -----------------------------------------------------------------------------------

# lakukan prediksi dengan data uji
y_pred = classifier.predict(x_test)

# cetak classification report
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------------------
# 6)                                 OPTIMIZATION
# -----------------------------------------------------------------------------------

# variabel error untuk menyimpan data rata-rata error
error = []

# melakukan perulangan untuk nilai k antara 1-40
for i in range(1, 40):
    # membuat classifier dengan k = i
    knn = KNeighborsClassifier(n_neighbors=i)

    # melakukan training dengan dataset
    knn.fit(x_train, y_train)

    # melakukan prediksi dengan data uji
    pred_i = knn.predict(x_test)

    # menghitung rata-rata error untuk hasil uji
    error.append(np.mean(pred_i != y_test))

# membuat plot baru hubungan antara nilai k dan rata-rata error prediksi
pyplot.figure(figsize=(12, 6))
pyplot.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
pyplot.title('Rata-Rata Error terhadap nilai K')
pyplot.xlabel('Nilai K')
pyplot.ylabel('Rata-Rata Error')
pyplot.show()

# -----------------------------------------------------------------------------------
# 7)                              MODEL PRESISTANCE
# -----------------------------------------------------------------------------------

# simpan model yang sudah di train dan evaluate
pickle.dump(classifier, open("iris-classification.model", "wb"))

# simpan juga scaler untuk menskala input data yang lain
pickle.dump(scaler, open("iris-scaler.model", "wb"))