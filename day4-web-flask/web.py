# import pickle untuk membaca model yang disimpan
import pickle

# import sklearn untuk menggunakan algoritma KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# import Flask untuk membuat web server
from flask import Flask, render_template, request

# buat objek Flask sebagai web server
app = Flask(__name__, static_folder="assets")

# membaca model yang sudah disimpan sebelumnya
scaler: StandardScaler = pickle.load(open("iris-scaler.model", 'rb'))
classifier: KNeighborsClassifier = pickle.load(open("iris-classification.model", 'rb'))

# RUTE HOME (/) - Ini adalah rute saat mengakses root website.
@app.route("/")
def home():
    # render HTML
    return render_template("home.html")

# RUTE PREDICT (/predict) - Ini adalah rute saat user men-submit
# data melalui form untuk melakukan prediksi
@app.route("/predict", methods=["POST"])
def predict():
    # membaca input dari form HTML
    x = [[float(request.form["sepal-length"]), float(request.form["sepal-width"]), float(request.form["petal-length"]), float(request.form["petal-width"])]]

    # men-skala nilai input
    x_scaled = scaler.transform(x)

    # melakukan prediksi
    result = classifier.predict(x_scaled)

    # render HTML
    return render_template("prediction.html", raw=x[0], scaled=x_scaled[0], result=result[0])

# mulai web server
if __name__ == "__main__":
    app.run(debug=True)