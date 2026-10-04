import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

CSV_DE_ANTRENAREE = "train.csv"
CSV_DE_TESTARE = "test.csv"
FOLDER_ANTRE = "train"
FOLDER_TESTARE = "test"
CSV_PT_OUTPUT = "submission_cnn.csv"
CSV_PT_OUTPUT_2 = "submission_ensemble.csv"

DIMENSIUNI_POZE = 64
DIMENSIUNI_BATCHY = 32
NR_EPOCI = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
RABDARE = 15
VAL_SPLIT = 0.20
NR_CLASE = 5
SPORURI_PT_ASAMBLARE = [42, 11, 121]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seteaza_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DataSetDeSemnaleLuminoase(Dataset):
    def __init__(self, iduri, taguri=None, folder_poze=FOLDER_ANTRE, augment=False):
        self.iduri = iduri
        self.taguri = taguri
        self.folder_poze = folder_poze
        self.augment = augment
        self.train_transform = transforms.Compose([
            transforms.Resize((DIMENSIUNI_POZE, DIMENSIUNI_POZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.val_transform = transforms.Compose([
            transforms.Resize((DIMENSIUNI_POZE, DIMENSIUNI_POZE)),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __len__(self): return len(self.iduri)

    def __getitem__(self, idx):
        calea_pozei = os.path.join(self.folder_poze, self.iduri[idx])
        #poza = Image.open(calea_pozei).convert("L")  # grayscale (incercare veche)
        poza = Image.open(calea_pozei).convert("RGB")
        transform = self.train_transform if self.augment else self.val_transform
        tensor_pt_pozaa = transform(poza)
        if self.taguri is not None:
            tag = int(self.taguri[idx]) - 1
            return tensor_pt_pozaa, tag
        else: return tensor_pt_pozaa


class CNNSemnaleRadioDeInaltaTensiune(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNSemnaleRadioDeInaltaTensiune, self).__init__()

        self.trasatui = nn.Sequential(
            #1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            #2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            #3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        self.clasificator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def redirectioneaza(self, x):
        x = self.trasatui(x)
        x = self.clasificator(x)
        return x


def anternare_per_epoca(model, incarcator_usbc, optimizator, criteriu_de_org):
    model.train()
    lossu_total, nimerit, total = 0.0, 0, 0
    for pozeee, taguri in incarcator_usbc:
        pozeee, taguri = pozeee.to(DEVICE), taguri.to(DEVICE)
        optimizator.zero_grad()
        outputs = model(pozeee)
        loss = criteriu_de_org(outputs, taguri)
        loss.backward()
        optimizator.step()
        lossu_total += loss.item() * pozeee.size(0)
        predictii_bune_sper = outputs.argmax(dim=1)
        nimerit += (predictii_bune_sper == taguri).sum().item()
        total += pozeee.size(0)
    return lossu_total/total, nimerit/total


def evaluarea_competentelor(model, incarcator_usbc, criteriu_de_org):
    model.eval()
    lossu_total, nimerit, total = 0.0, 0, 0
    toate_predictiile_inapoi, toate_tintele_inainte = [], []
    with torch.no_grad():
        for pozeee, taguri in incarcator_usbc:
            pozeee, taguri = pozeee.to(DEVICE), taguri.to(DEVICE)
            outputs = model(pozeee)
            loss = criteriu_de_org(outputs, taguri)
            lossu_total  += loss.item() * pozeee.size(0)
            predictii_bune_sper = outputs.argmax(dim=1)
            nimerit += (predictii_bune_sper == taguri).sum().item()
            total += pozeee.size(0)
            toate_predictiile_inapoi.extend(predictii_bune_sper.cpu().numpy())
            toate_tintele_inainte.extend(taguri.cpu().numpy())
    return lossu_total / total, nimerit / total, toate_predictiile_inapoi, toate_tintele_inainte

#varianta 1 cu majority vote
# def predictie(model, incarcator_usbc):
#     model.eval()
#     toate_predictiile_inapoi = []
#     with torch.no_grad():
#         for pozeee in incarcator_usbc:
#             pozeee = pozeee.to(DEVICE)
#             output = model(pozeee)
#             predictii_bune_sper = output.argmax(dim=1)
#             toate_predictiile_inapoi.extend(predictii_bune_sper.cpu().numpy())
#     return np.array(toate_predictiile_inapoi)
def predictie(model, incarcator_usbc):
    import torch.nn.functional as F
    model.eval()
    totalitatea_probabilitatilor = []
    with torch.no_grad():
        for pozeee in incarcator_usbc:
            pozeee = pozeee.to(DEVICE)
            logits = model(pozeee)
            probabilitati = F.softmax(logits, dim=1)
            totalitatea_probabilitatilor.append(probabilitati.cpu().numpy())
    return np.concatenate(totalitatea_probabilitatilor, axis=0)


def matricea_de_confuion(y_true, y_pred, title="Confusion Matrix"):
    matr = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matr, annot=True, fmt="d", cmap="Oranges",xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Matricea confusion salvata: {fname}")


def grafice_verificare(loss_pe_antrenare, loss_pe_validr, acuratetea_pe_train, acuratetea_pe_val, title="Training"):
    epoci = range(1, len(loss_pe_antrenare) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epoci, loss_pe_antrenare, label="Train Loss")
    ax1.plot(epoci, loss_pe_validr, label="Val Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epoci, acuratetea_pe_train, label="Train Acc")
    ax2.plot(epoci, acuratetea_pe_val,   label="Val Acc")
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_curves.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Diagrame sugestive: {fname}")


def antrenarea_modelului_incepator(incarcare_antre, incarcare_valid, seed=42, tag="Model"):
    seteaza_seed(seed)
    model = CNNSemnaleRadioDeInaltaTensiune(num_classes=NR_CLASE).to(DEVICE)
    criteriu_de_org = nn.CrossEntropyLoss()
    optimizator = optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizator, mode="min", factor=0.5, patience=5
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizator, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    max_acuratete_validr = 0.0
    cel_mai_bun_model_wts = copy.deepcopy(model.state_dict())
    imbunatatiti_none = 0
    loss_pe_antrenare, loss_pe_validr = [], []
    acuratetea_pe_train,acuratetea_pe_val   = [], []
    print(f"Antrenare {tag} (seed={seed})")
    for epoca in range(1, NR_EPOCI + 1):
        lossu_antre_epoca, acuratete_antre_epoca = anternare_per_epoca(model, incarcare_antre, optimizator, criteriu_de_org)
        lossu_validr_epoca, acuratete_validr_epoca, _, _ = evaluarea_competentelor(model, incarcare_valid, criteriu_de_org)
        scheduler.step(lossu_validr_epoca)
        loss_pe_antrenare.append(lossu_antre_epoca)
        loss_pe_validr.append(lossu_validr_epoca)
        acuratetea_pe_train.append(acuratete_antre_epoca)
        acuratetea_pe_val.append(acuratete_validr_epoca)
        print(
            f"Epoch {epoca:3d}/{NR_EPOCI}  "
            f"TLoss: {lossu_antre_epoca:.4f}  TAcc: {acuratete_antre_epoca*100:.2f}%  "
            f"VLoss: {lossu_validr_epoca:.4f}  VAcc: {acuratete_validr_epoca*100:.2f}%  "
            f"LR: {optimizator.param_groups[0]['lr']:.2e}")
        if acuratete_validr_epoca > max_acuratete_validr:
            max_acuratete_validr = acuratete_validr_epoca
            cel_mai_bun_model_wts = copy.deepcopy(model.state_dict())
            imbunatatiti_none = 0
        else:
            imbunatatiti_none += 1
            if imbunatatiti_none >= RABDARE:
                print(f"Nu se mai invata nimic de la epoca {epoca}.")
                break

    model.load_state_dict(cel_mai_bun_model_wts)
    print(f"\nCea mai buna acuratete({tag}): {max_acuratete_validr*100:.2f}%")
    istoric_n = {
        "loss_pe_antrenare": loss_pe_antrenare,
        "loss_pe_validr":   loss_pe_validr,
        "acuratetea_pe_train":   acuratetea_pe_train,
        "acuratetea_pe_val":     acuratetea_pe_val,
    }
    return model, istoric_n


def main():
    seteaza_seed(42)
    antre_df = pd.read_csv(CSV_DE_ANTRENAREE)
    test_df  = pd.read_csv(CSV_DE_TESTARE)
    toti_id = antre_df["id"].values
    toate_tagurile = antre_df["label"].values
    test_ids = test_df["id"].values

    n_total = len(toti_id)
    n_val = int(n_total * VAL_SPLIT)
    permutarea = np.random.permutation(n_total)
    index_validr = permutarea[:n_val]
    index_antre = permutarea[n_val:]
    iduri_antre = toti_id[index_antre]
    hashtag_antre_taguri = toate_tagurile[index_antre]
    iduri_validr = toti_id[index_validr]
    hashtag_validr_taguri = toate_tagurile[index_validr]
    print(f"Train: {len(iduri_antre)} | Val: {len(iduri_validr)} | Test: {len(test_ids)}")

    antei_ds = DataSetDeSemnaleLuminoase(iduri_antre, hashtag_antre_taguri, FOLDER_ANTRE, augment=True)
    val_ds = DataSetDeSemnaleLuminoase(iduri_validr,hashtag_validr_taguri,FOLDER_ANTRE, augment=False)
    validr_ds = DataSetDeSemnaleLuminoase(test_ids,None,FOLDER_TESTARE,  augment=False)
    incarcare_antre = DataLoader(antei_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=True,  num_workers=0)
    incarcare_valid = DataLoader(val_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=False, num_workers=0)
    incarcare_test = DataLoader(validr_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=False, num_workers=0)

    #faza1: 08/20
    model, istoric_n = antrenarea_modelului_incepator(incarcare_antre, incarcare_valid, seed=42, tag="CNN Faza1")
    grafice_verificare(
        istoric_n["loss_pe_antrenare"], istoric_n["loss_pe_validr"],
        istoric_n["acuratetea_pe_train"],   istoric_n["acuratetea_pe_val"],
        title="CNN Faza1"
    )

    criteriu_de_org = nn.CrossEntropyLoss()
    _, validr_acura, validr_predictii, validr_tinte = evaluarea_competentelor(model, incarcare_valid, criteriu_de_org)
    print(f"\nAcuratetea finala: {validr_acura*100:.2f}%")

    predictii_validr_shiftate_mai_in_colo   = [p + 1 for p in validr_predictii]
    tinte_validr_shiftate_mai_in_colo = [t + 1 for t in validr_tinte]
    matricea_de_confuion(tinte_validr_shiftate_mai_in_colo, predictii_validr_shiftate_mai_in_colo,title="CNN Phase1 Validation Confusion Matrix")

    #faza2: 100
    print("faza2")
    all_in_training_session_ds= DataSetDeSemnaleLuminoase(toti_id, toate_tagurile, FOLDER_ANTRE, augment=True)
    incarcare_full = DataLoader(all_in_training_session_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=True, num_workers=0)
    tot_modelul_la_atac, hist_full = antrenarea_modelului_incepator(incarcare_full, incarcare_valid, seed=42, tag="CNN Phase2 Full")
    # predictii_test_fr = predictie(tot_modelul_la_atac, incarcare_test) + 1  #eroare la trecerea rgb
    predictii_test_fr = predictie(tot_modelul_la_atac, incarcare_test).argmax(axis=1) + 1
    pd.DataFrame({"id": test_ids, "label": predictii_test_fr}).to_csv(CSV_PT_OUTPUT, index=False)
    print(f"\nSalvare:  {CSV_PT_OUTPUT}")

    #asamblare
    # predictii_de_asamblare = []
    # for seed in SPORURI_PT_ASAMBLARE:
    #     m, _ = antrenarea_modelului_incepator(incarcare_full, incarcare_valid, seed=seed,
    #                        tag=f"Ensemble seed={seed}")
    #     predictii_bune_sper = predictie(m, incarcare_test)
    #     predictii_de_asamblare.append(predictii_bune_sper)
    predictii_de_asamblare = []
    for seed in SPORURI_PT_ASAMBLARE:
        m, _ = antrenarea_modelului_incepator(incarcare_full, incarcare_valid, seed=seed,tag=f"Ensemble seed={seed}")
        probabilitati = predictie(m, incarcare_test)
        predictii_de_asamblare.append(probabilitati)

    #maj vote
    # predictii_de_asamblare = np.stack(predictii_de_asamblare, axis=1)
    # predictii_finale = np.apply_along_axis(
    #     lambda x: np.bincount(x, minlength=NR_CLASE).argmax(),
    #     axis=1, arr=predictii_de_asamblare
    # ) + 1    # shift back to 1–5
    media_probabilitatilor = np.mean(predictii_de_asamblare, axis=0)
    predictii_finale = media_probabilitatilor.argmax(axis=1) + 1
    pd.DataFrame({"id": test_ids, "label": predictii_finale}).to_csv(CSV_PT_OUTPUT_2, index=False)
    print(f"Asamblare salv {CSV_PT_OUTPUT_2}")


if __name__ == "__main__":
    main() #max pe incercari diverse ~58%
