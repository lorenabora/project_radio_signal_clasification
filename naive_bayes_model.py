import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


CSV_DE_ANTRENAREEE = "train.csv"
CSV_DE_TESTARE= "test.csv"
FOLDER_ANTRE = "train"
FOLDER_TESTARE = "test"
CSV_PT_OUTPUT = "submission_nb.csv"

DIMENSIUNE_IMAG = (64, 64)
VAL_SPLIT = 0.20
RANDOM_SEED = 42


def incarcare_imagini(path):
    poza = Image.open(path).convert("L")
    poza = poza.resize(DIMENSIUNE_IMAG)
    return np.array(poza, dtype=np.float32)


def extrage_trasaturi(poza_array):
    #normalizare pixeli 01
    flat = poza_array.flatten() / 255.0
    trasaturi_hog = hog(poza_array,orientations=8,pixels_per_cell=(8, 8),cells_per_block=(2, 2),feature_vector=True)
    statistici = np.array([
        poza_array.mean(), poza_array.std(), poza_array.min(), poza_array.max(),
        np.percentile(poza_array, 25), np.percentile(poza_array, 50),
        np.percentile(poza_array, 75), np.percentile(poza_array, 90),
    ], dtype=np.float32)
    #concatenare intr-un vector
    return np.concatenate([flat, trasaturi_hog, statistici])


def incarca_dataset(csv_path, poza_dir, tag_tu_esti=True):
    df = pd.read_csv(csv_path)
    trasaturi = []
    taguri = [] if tag_tu_esti else None
    ids = []
    print(f"Loading {'training' if tag_tu_esti else 'test'} data from '{poza_dir}'...")
    for _, row in df.iterrows():
        poza_path = os.path.join(poza_dir, row["id"])
        poza = incarcare_imagini(poza_path)
        feats = extrage_trasaturi(poza)
        trasaturi.append(feats)
        ids.append(row["id"])
        if tag_tu_esti:
            taguri.append(int(row["label"]))

    X = np.array(trasaturi, dtype=np.float32)
    y = np.array(taguri, dtype=np.int32) if tag_tu_esti else None
    print(f"  Loaded {len(X)} samples | Feature vector size: {X.shape[1]}")
    return X, y, ids



def matrice_de_cofuion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Confusion matrix saved: {fname}")
    return cm


def main():
    np.random.seed(RANDOM_SEED)

    X_all, y_all, _ = incarca_dataset(CSV_DE_ANTRENAREEE, FOLDER_ANTRE, tag_tu_esti=True)
    #80/20
    n_total = len(X_all)
    n_val = int(n_total * VAL_SPLIT)

    indicu = np.random.permutation(n_total)
    val_idx = indicu[:n_val]
    train_idx = indicu[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"\nSplit - Train: {len(X_train)} | Val: {len(X_val)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    naiv_bay = GaussianNB()
    naiv_bay.fit(X_train, y_train)

    val_preds = naiv_bay.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\nValidation Accuracy (Naive Bayes): {val_acc * 100:.2f}%")

    matrice_de_cofuion(y_val, val_preds, title="Naive Bayes Validation Confusion Matrix")

    X_all_scaled = scaler.fit_transform(X_all)
    naiv_bay.fit(X_all_scaled, y_all)
    train_preds = naiv_bay.predict(X_all_scaled)
    train_acc = accuracy_score(y_all, train_preds)
    print(f"Full Training Accuracy: {train_acc * 100:.2f}%")

    X_test, _, test_ids = incarca_dataset(CSV_DE_TESTARE, FOLDER_TESTARE, tag_tu_esti=False)
    X_test_scaled = scaler.transform(X_test)
    test_preds = naiv_bay.predict(X_test_scaled)

    submission = pd.DataFrame({"id": test_ids, "label": test_preds})
    submission.to_csv(CSV_PT_OUTPUT, index=False)
    print(f"Submission saved: {CSV_PT_OUTPUT}")
    print(submission.head(10))


if __name__ == "__main__":
    main()
