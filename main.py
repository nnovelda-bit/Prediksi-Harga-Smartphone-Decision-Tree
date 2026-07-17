# =====================================================
# FINAL PROJECT PENGANTAR SAINS DATA
# Prediksi Harga Smartphone Menggunakan Decision Tree
# =====================================================

from utils import *


def main():

    print("=" * 60)
    print("      FINAL PROJECT PENGANTAR SAINS DATA")
    print("  PREDIKSI HARGA SMARTPHONE DECISION TREE")
    print("=" * 60)

    # =================================================
    # Membaca Dataset
    # =================================================
    print("\nMembaca dataset...")

    data = load_data("dataset/train.csv")

    print("Dataset berhasil dibaca.\n")

    # =================================================
    # Informasi Dataset
    # =================================================
    informasi_data(data)

    # =================================================
    # Menampilkan Data
    # =================================================
    tampil_data(data)

    # =================================================
    # Missing Value
    # =================================================
    cek_missing_value(data)

    # =================================================
    # Statistik Dataset
    # =================================================
    statistik_dataset(data)

    # =================================================
    # Menjalankan Machine Learning
    # =================================================
    print("\nMemulai proses Machine Learning...\n")

    accuracy = proses_machine_learning(data)

    # =================================================
    # Hasil Akhir
    # =================================================
    print("\n" + "=" * 60)
    print("PROJECT BERHASIL DIJALANKAN")
    print("=" * 60)

    print(f"Accuracy Model : {accuracy:.4f}")

    print("\nFile yang berhasil dibuat:")
    print("- hasil_analisis.txt")
    print("- output/grafik_output.png")
    print("- output/confusion_matrix.png")
    print("- output/decision_tree.png")

    print("\nTerima kasih telah menggunakan program ini.")


# =====================================================
# Menjalankan Program
# =====================================================
if __name__ == "__main__":
    main()