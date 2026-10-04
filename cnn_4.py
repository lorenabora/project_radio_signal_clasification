import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
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

DIMENSIUNI_POZE = None
DIMENSIUNI_BATCHY = 32
NR_EPOCI = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
RABDAREE = 15
VAL_SPLIT = 0.20
NR_CLASE = 5
SPORURI_PT_ASAMBLARE = [42, 11, 121]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


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
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.iduri)

    def __getitem__(self, idx):
        poza = Image.open(
            os.path.join(self.folder_poze, self.iduri[idx])
        ).convert("L")
        tensor_pt_pozaa = (self.train_transform if self.augment
                      else self.val_transform)(poza)
        if self.taguri is not None:
            return tensor_pt_pozaa, int(self.taguri[idx]) - 1
        return tensor_pt_pozaa


class CNNSemnaleRadioDeInaltaTensiune(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNSemnaleRadioDeInaltaTensiune, self).__init__()

        self.trasaturi = nn.Sequential(
            #1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            #2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            #3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            #4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            #5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.clasificator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 1 * 512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def redirectioneaza(self, x):
        return self.clasificator(self.trasaturi(x))



def anternare_per_epoca(model, incarcator_usbc, optimizator, criteriu_de_org, scheduler):
    model.train()
    lossu_total, nimerit, total = 0.0, 0, 0
    for pozeee, taguri in incarcator_usbc:
        pozeee, taguri = pozeee.to(DEVICE), taguri.to(DEVICE)
        optimizator.zero_grad()
        out = model(pozeee)
        loss = criteriu_de_org(out, taguri)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #nou, Nu, spalte cu pervol
        optimizator.step()
        #adaugat odata cu scheduler
        scheduler.step()
        lossu_total += loss.item() * pozeee.size(0)
        nimerit += (out.argmax(1) == taguri).sum().item()
        total += pozeee.size(0)
    return lossu_total / total, nimerit / total


def evaluarea_competentelor(model, incarcator_usbc, criteriu_de_org):
    model.eval()
    lossu_total, nimerit, total = 0.0, 0, 0
    toate_predictiile_inapoi, toate_tintele_inainte = [], []
    with torch.no_grad():
        for pozeee, taguri in incarcator_usbc:
            pozeee, taguri = pozeee.to(DEVICE), taguri.to(DEVICE)
            out = model(pozeee)
            loss = criteriu_de_org(out, taguri)
            lossu_total += loss.item() * pozeee.size(0)
            predictii_buna_sper = out.argmax(1)
            nimerit += (predictii_buna_sper == taguri).sum().item()
            total += pozeee.size(0)
            toate_predictiile_inapoi.extend(predictii_buna_sper.cpu().numpy())
            toate_tintele_inainte.extend(taguri.cpu().numpy())
    return lossu_total / total, nimerit / total, toate_predictiile_inapoi, toate_tintele_inainte


def predictie_1(model, incarcator_usbc):
    import torch.nn.functional as F
    model.eval()
    #tta
    # iterare 1/variant
    varianti_pt_tta = [ "original", "hflip", "vflip", "hvflip",]
    probabilitatile_var = {v: [] for v in varianti_pt_tta}
    with torch.no_grad():
        for pozeee in incarcator_usbc:
            pozeee = pozeee.to(DEVICE)
            #mr.original
            probabilitatile_var["original"].append(F.softmax(model(pozeee), dim=1).cpu().numpy())
            #orizontala
            pozee_hor = torch.flip(pozeee, dims=[3])
            probabilitatile_var["hflip"].append(F.softmax(model(pozee_hor), dim=1).cpu().numpy())
            #verti
            pozee_verti = torch.flip(pozeee, dims=[2])
            probabilitatile_var["vflip"].append(F.softmax(model(pozee_verti), dim=1).cpu().numpy())
            #combinare flips
            combo_poze = torch.flip(pozeee, dims=[2, 3])
            probabilitatile_var["hvflip"].append(F.softmax(model(combo_poze), dim=1).cpu().numpy())
    clatite = np.stack([np.concatenate(probabilitatile_var[v], axis=0) for v in varianti_pt_tta],axis=0)
    return clatite.mean(axis=0)


def matricea_de_confuion(y_true, y_pred, title="Confusion Matrix"):
    matr = confusion_matrix(y_true, y_pred)
    norm_matr = matr.astype(float) / matr.sum(axis=1, keepdims=True) * 100
    adnotare = np.array([[f"{matr[i,j]}\n({norm_matr[i,j]:.1f}%)"
                         for j in range(5)] for i in range(5)])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(norm_matr, annot=adnotare, fmt="", cmap="Oranges", xticklabels=range(1, 6), yticklabels=range(1, 6), linewidths=0.5, vmin=0, vmax=100, ax=ax,annot_kws={"size": 10})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Confusion matrix saved: {fname}")


def grafice_verificare(loss_pe_antrenare, loss_pe_validari, acuratete_pe_train, acuratete_pe_val, title="Training"):
    epoci = range(1, len(loss_pe_antrenare) + 1)
    best_out_of_best_epoca = int(np.argmax(acuratete_pe_val)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title} — Train vs Validation", fontsize=13,fontweight="bold")

    ax1.plot(epoci, loss_pe_antrenare, linewidth=2, label="Train Loss",color="#4C72B0")
    ax1.plot(epoci, loss_pe_validari, linewidth=2, label="Val Loss",color="#C44E52", linestyle="--")
    ax1.axvline(best_out_of_best_epoca, color="green", linestyle=":", linewidth=1.5,label=f"Best epoch {best_out_of_best_epoca}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.35)

    ax2.plot(epoci, [a * 100 for a in acuratete_pe_train], linewidth=2,label="Train Acc", color="#55A868")
    ax2.plot(epoci, [a * 100 for a in acuratete_pe_val], linewidth=2,label="Val Acc", color="#DD8452", linestyle="--")
    ax2.fill_between(epoci,[a * 100 for a in acuratete_pe_train],[a * 100 for a in acuratete_pe_val],alpha=0.10, color="gray", label="Train-Val gap")
    ax2.axvline(best_out_of_best_epoca, color="green", linestyle=":", linewidth=1.5,label=f"Best val: {max(acuratete_pe_val)*100:.2f}%")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.35)

    plt.tight_layout()
    fname = f"{title.lower().replace(' ', '_')}_curves.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Training curves saved: {fname}")



def experimente_pe_hiperpara(incarcare_antre, incarcare_valid):
    configuratii = [
        {"lr": 1e-2, "wd": 1e-4, "tag": "LR=1e-2"},
        {"lr": 1e-3, "wd": 1e-4, "tag": "LR=1e-3 (default)"},
        {"lr": 5e-4, "wd": 1e-4, "tag": "LR=5e-4"},
        {"lr": 1e-4, "wd": 1e-4, "tag": "LR=1e-4"},
        {"lr": 1e-3, "wd": 1e-3, "tag": "WD=1e-3"},
    ]
    rez = []
    for cfg in configuratii:
        seteaza_seed(42)
        m = CNNSemnaleRadioDeInaltaTensiune(NR_CLASE).to(DEVICE)
        crt = nn.CrossEntropyLoss()
        opt = optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        # sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        sch = optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg["lr"], steps_per_epoch=len(incarcare_antre), epochs=15)
        cel_mai_bun, imbunatatiti_none = 0.0, 0
        for _ in range(15):
            anternare_per_epoca(m, incarcare_antre, opt, crt, sch)
            lossu_validr, acuratetea_validr, _, _ = evaluarea_competentelor(m, incarcare_valid, crt)
            # sch.step(lossu_validr)
            if acuratetea_validr > cel_mai_bun:
                cel_mai_bun = acuratetea_validr; imbunatatiti_none = 0
            else:
                imbunatatiti_none += 1
                if imbunatatiti_none >= 5: break
        rez.append({"Config": cfg["tag"], "Val Acc (%)": f"{cel_mai_bun*100:.2f}"})
        print(f"  {cfg['tag']:28s} → {cel_mai_bun*100:.2f}%")
    df = pd.DataFrame(rez)
    df.to_csv("hyperparameter_results.csv", index=False)
    print("Salvat: hyperparameter_results.csv")
    return df


def antrenarea_modelulului_incepator(incarcare_antre, incarcare_valid, seed=42, tag="Model",class_weights=None):
    seteaza_seed(seed)
    model = CNNSemnaleRadioDeInaltaTensiune(num_classes=NR_CLASE).to(DEVICE)
    criteriu_de_org = nn.CrossEntropyLoss(weight=class_weights)
    optimizator = optim.Adam(model.parameters(),lr=LR, weight_decay=WEIGHT_DECAY)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizator, mode="min", factor=0.5, patience=5, min_lr=1e-6) #pot schimba la 5 patiance
    #OneCycleLR funcționează pe baza numărului total de pași
    scheduler = optim.lr_scheduler.OneCycleLR(optimizator,max_lr=1e-3,steps_per_epoch=len(incarcare_antre), epochs=NR_EPOCI)

    max_acuratete_valid = 0.0
    cel_mai_bun_model_wts = copy.deepcopy(model.state_dict())
    imbunatatiri_none = 0
    loss_pe_antrenare, loss_pe_validari, acuratete_pe_train, acuratete_pe_val = [], [], [], []
    print(f"\n{'='*55}\nTraining {tag} (seed={seed})\n{'='*55}")
    for epoca in range(1, NR_EPOCI + 1):
        # warmup pt reduceLROnPlateau
        # if epoca <= 3:
        #     lr_scale = epoca / 3
        #     for param_group in optimizator.param_groups:
        #         param_group['lr'] = LR * lr_scale

        # lossu_antre_epoca, acurateate_antre_epoca = anternare_per_epoca(model, incarcare_antre, optimizator, criteriu_de_org)
        lossu_antre_epoca, acurateate_antre_epoca = anternare_per_epoca(model, incarcare_antre, optimizator, criteriu_de_org, scheduler)
        lossu_validr, acuratetea_validr, _, _ = evaluarea_competentelor(model, incarcare_valid, criteriu_de_org)
        # scheduler.step(lossu_validr)

        loss_pe_antrenare.append(lossu_antre_epoca); loss_pe_validari.append(lossu_validr)
        acuratete_pe_train.append(acurateate_antre_epoca);    acuratete_pe_val.append(acuratetea_validr)

        print(f"Ep {epoca:3d}/{NR_EPOCI}  "
              f"TrLoss:{lossu_antre_epoca:.4f}  TrAcc:{acurateate_antre_epoca*100:.2f}%  "
              f"VlLoss:{lossu_validr:.4f}  VlAcc:{acuratetea_validr*100:.2f}%  "
              f"LR:{optimizator.param_groups[0]['lr']:.1e}")

        if acuratetea_validr > max_acuratete_valid:
            max_acuratete_valid   = acuratetea_validr
            cel_mai_bun_model_wts = copy.deepcopy(model.state_dict())
            imbunatatiri_none     = 0
        else:
            imbunatatiri_none += 1
            if imbunatatiri_none >= RABDAREE:
                print(f"  Early stopping at epoch {epoca}.")
                break

    model.load_state_dict(cel_mai_bun_model_wts)
    print(f"Best Val Acc ({tag}): {max_acuratete_valid*100:.2f}%")
    return model, {"loss_pe_antrenare": loss_pe_antrenare, "loss_pe_validari": loss_pe_validari, "acuratete_pe_train":   acuratete_pe_train,   "acuratete_pe_val":   acuratete_pe_val}



def main():
    seteaza_seed(42)
    antre_df = pd.read_csv(CSV_DE_ANTRENAREE)
    test_df = pd.read_csv(CSV_DE_TESTARE)
    toti_id = antre_df["id"].values
    toate_tagurile = antre_df["label"].values
    test_ids = test_df["id"].values

    #impartirea 80/20
    permutarea = np.random.permutation(len(toti_id))
    index_validr = permutarea[:int(len(toti_id) * VAL_SPLIT)]
    index_antre = permutarea[int(len(toti_id) * VAL_SPLIT):]

    iduri_antre = toti_id[index_antre]; train_labels = toate_tagurile[index_antre]
    iduri_validr = toti_id[index_validr]; val_labels = toate_tagurile[index_validr]
    print(f"Train:{len(iduri_antre)} | Val:{len(iduri_validr)} | Test:{len(test_ids)}")

    #dupa primele rulari am sesizat ca in clasa 4 se creaza multe confuzii
    #asadar am adaugat "greutati" => au imbunatatit major modelul
    numar_clasa = np.bincount(train_labels,minlength=6)[1:]
    greutati = 1.0 / (numar_clasa.astype(float) + 1e-6)
    greutati = greutati / greutati.sum() * NR_CLASE
    greutati_tensor = torch.FloatTensor(greutati).to(DEVICE)
    print("Class weights:",{i + 1: f"{w:.3f}" for i, w in enumerate(greutati)})

    #incarcare date
    antei_ds = DataSetDeSemnaleLuminoase(iduri_antre, train_labels, FOLDER_ANTRE, augment=True)
    validr_ds = DataSetDeSemnaleLuminoase(iduri_validr,val_labels, FOLDER_ANTRE, augment=False)
    test_ds = DataSetDeSemnaleLuminoase(test_ids,None,FOLDER_TESTARE,  augment=False)

    incarcare_antre = DataLoader(antei_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=True,  num_workers=0)
    incarcare_valid = DataLoader(validr_ds,batch_size=DIMENSIUNI_BATCHY, shuffle=False, num_workers=0)
    incarcare_test  = DataLoader(test_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=False, num_workers=0)

    #kyperparametrii checked
    # experimente_pe_hiperpara(incarcare_antre, incarcare_valid)

    #faza 1
    model, istoric_obscur = antrenarea_modelulului_incepator(incarcare_antre, incarcare_valid,seed=42, tag="CNN Phase1",class_weights=greutati_tensor)
    grafice_verificare(istoric_obscur["loss_pe_antrenare"], istoric_obscur["loss_pe_validari"],istoric_obscur["acuratete_pe_train"],   istoric_obscur["acuratete_pe_val"],title="CNN Phase1")

    criteriu_de_org = nn.CrossEntropyLoss(weight=greutati_tensor)
    _, validr_accura, validr_predictii, validr_tinte = evaluarea_competentelor(model, incarcare_valid, criteriu_de_org)
    print(f"\nFinal Validation Accuracy (Phase 1): {validr_accura*100:.2f}%")

    matricea_de_confuion([t + 1 for t in validr_tinte],[p + 1 for p in validr_predictii],title="CNN Phase1 Validation Confusion Matrix")

    #faza 2
    all_in_la_atac_ds = DataSetDeSemnaleLuminoase(toti_id, toate_tagurile, FOLDER_ANTRE, augment=True)
    incarcare_totala = DataLoader(all_in_la_atac_ds, batch_size=DIMENSIUNI_BATCHY, shuffle=True, num_workers=0)
    model_final, _ = antrenarea_modelulului_incepator(incarcare_totala, incarcare_valid,seed=42, tag="CNN Phase2 Full",class_weights=greutati_tensor)

    predictii_testing_sgl = predictie_1(model_final, incarcare_test).argmax(axis=1) + 1
    pd.DataFrame({"id": test_ids, "label": predictii_testing_sgl}).to_csv(CSV_PT_OUTPUT, index=False)
    print(f"Single-model submission: {CSV_PT_OUTPUT}")

    # asamblare+ softvot
    probabilitati_asamblare = []
    for seed in SPORURI_PT_ASAMBLARE:
        m, _ = antrenarea_modelulului_incepator(incarcare_totala, incarcare_valid, seed=seed,tag=f"Ensemble seed={seed}",class_weights=greutati_tensor)
        probabilitati_asamblare.append(predictie_1(m, incarcare_test))

    medie_probabilitati = np.mean(probabilitati_asamblare, axis=0)
    predictii_finale = medie_probabilitati.argmax(axis=1) + 1

    pd.DataFrame({"id": test_ids, "label": predictii_finale}).to_csv(CSV_PT_OUTPUT_2, index=False)
    print(f"Asamblare subm: {CSV_PT_OUTPUT_2}")


if __name__ == "__main__":
    main() #maxim moentan ~66%