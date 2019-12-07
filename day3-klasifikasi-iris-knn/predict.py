# import argparse untuk menerima input dari command line
import argparse

# import pickle untuk membaca model yang disimpan
import pickle

# import sklearn untuk menggunakan algoritma KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# membuat aturan input dari command line
parser = argparse.ArgumentParser(description="Prediksi IRIS")
parser.add_argument("sl", metavar="sepal-length", type=float)
parser.add_argument("sw", metavar="sepal-width", type=float)
parser.add_argument("pl", metavar="petal-length", type=float)
parser.add_argument("pw", metavar="petal-width", type=float)

# melakukan parsing/membaca input dari command line
args = parser.parse_args()

# membaca model yang sudah disimpan sebelumnya
scaler: StandardScaler = pickle.load(open("iris-scaler.model", 'rb'))
classifier: KNeighborsClassifier = pickle.load(open("iris-classification.model", 'rb'))

# membaca input dari command line berdasarkan aturan yang sudah dibuat
x = [[args.sl, args.sw, args.pl, args.pw]]
print("Data mentah:")
print("  sepal length: %f" % x[0][0])
print("  sepal width : %f" % x[0][1])
print("  petal length: %f" % x[0][2])
print("  petal width : %f" % x[0][3])
print()

# menskalakan input data
x = scaler.transform(x)
print("Data setelah di skala:")
print("  sepal length: %f" % x[0][0])
print("  sepal width : %f" % x[0][1])
print("  petal length: %f" % x[0][2])
print("  petal width : %f" % x[0][3])
print()

# melakukan prediksi
y_predict = classifier.predict(x)
print("Hasil prediksi: %s" % y_predict[0])