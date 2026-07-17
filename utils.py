# =====================================================
# FINAL PROJECT PENGANTAR SAINS DATA
# Prediksi Harga Smartphone Menggunakan Decision Tree
# =====================================================

# Import Library
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================
# Membaca Dataset
# =====================================================
def load_data(path):

    data = pd.read_csv(path)

    return data


# =====================================================
# Menampilkan Informasi Dataset
# =====================================================
def informasi_data(data):

    print("=" * 60)
    print("INFORMASI DATASET")
    print("=" * 60)

    print(data.info())


# =====================================================
# Menampilkan Lima Data Pertama
# =====================================================
def tampil_data(data):

    print("\n")
    print("=" * 60)
    print("LIMA DATA PERTAMA")
    print("=" * 60)

    print(data.head())


# =====================================================
# Mengecek Missing Value
# =====================================================
def cek_missing_value(data):

    print("\n")
    print("=" * 60)
    print("MISSING VALUE")
    print("=" * 60)

    print(data.isnull().sum())


# =====================================================
# Statistik Dataset
# =====================================================
def statistik_dataset(data):

    print("\n")
    print("=" * 60)
    print("STATISTIK DATASET")
    print("=" * 60)

    print(data.describe())

    return data.describe()


# =====================================================
# Membuat Grafik
# =====================================================
def buat_grafik(data):

    plt.figure(figsize=(7,5))

    data["price_range"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Distribusi Kategori Harga Smartphone")
    plt.xlabel("Price Range")
    plt.ylabel("Jumlah Data")

    plt.tight_layout()

    plt.savefig("output/grafik_output.png")

    plt.close()

    # =====================================================
# Melatih Model Decision Tree
# =====================================================
def train_model(data):

    print("\n")
    print("=" * 60)
    print("MEMBANGUN MODEL DECISION TREE")
    print("=" * 60)

    # Memisahkan fitur dan target
    X = data.drop("price_range", axis=1)
    y = data["price_range"]

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Membuat model
    model = DecisionTreeClassifier(random_state=42)

    # Melatih model
    model.fit(X_train, y_train)

    # Melakukan prediksi
    y_pred = model.predict(X_test)

    print("Model berhasil dibuat.")

    return model, X_test, y_test, y_pred


# =====================================================
# Evaluasi Model
# =====================================================
def evaluasi_model(model, X_test, y_test, y_pred):

    print("\n")
    print("=" * 60)
    print("HASIL EVALUASI MODEL")
    print("=" * 60)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")

    # Classification Report
    report = classification_report(y_test, y_pred)

    print("\nClassification Report")
    print(report)

    return accuracy, report


# =====================================================
# Confusion Matrix
# =====================================================
def tampil_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("output/confusion_matrix.png")

    plt.close()

    print("Confusion Matrix berhasil disimpan.")


# =====================================================
# Visualisasi Decision Tree
# =====================================================
def tampil_decision_tree(model):

    plt.figure(figsize=(20,10))

    plot_tree(
        model,
        filled=True,
        fontsize=6,
        rounded=True
    )

    plt.title("Visualisasi Decision Tree")

    plt.tight_layout()

    plt.savefig("output/decision_tree.png")

    plt.close()

    print("Visualisasi Decision Tree berhasil disimpan.")

    # =====================================================
# Menyimpan Hasil Analisis ke File TXT
# =====================================================
def simpan_hasil_txt(data, accuracy, report):

    with open("hasil_analisis.txt", "w", encoding="utf-8") as file:

        file.write("=" * 50 + "\n")
        file.write("HASIL ANALISIS DATASET SMARTPHONE\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Jumlah Data      : {len(data)}\n")
        file.write(f"Jumlah Kolom     : {len(data.columns)}\n\n")

        file.write("Nama Kolom:\n")

        for kolom in data.columns:
            file.write(f"- {kolom}\n")

        file.write("\n")

        file.write("=" * 50 + "\n")
        file.write("STATISTIK DATASET\n")
        file.write("=" * 50 + "\n\n")

        file.write(str(data.describe()))

        file.write("\n\n")

        file.write("=" * 50 + "\n")
        file.write("HASIL EVALUASI MODEL\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Accuracy : {accuracy:.4f}\n\n")

        file.write("Classification Report\n\n")

        file.write(report)

    print("\nHasil analisis berhasil disimpan ke hasil_analisis.txt")


# =====================================================
# Menjalankan Seluruh Proses
# =====================================================
def proses_machine_learning(data):

    # Membuat grafik distribusi
    buat_grafik(data)

    # Training model
    model, X_test, y_test, y_pred = train_model(data)

    # Evaluasi model
    accuracy, report = evaluasi_model(
        model,
        X_test,
        y_test,
        y_pred
    )

    # Membuat confusion matrix
    tampil_confusion_matrix(
        y_test,
        y_pred
    )

    # Membuat visualisasi decision tree
    tampil_decision_tree(model)

    # Menyimpan hasil analisis
    simpan_hasil_txt(
        data,
        accuracy,
        report
    )

    print("\nSemua proses Machine Learning selesai.")

    return accuracy