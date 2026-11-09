"""
Classification d'images médicales (radiographies...) : normal / pneumonie - 4 méthodes
(SVM + HOG, Random Forest + HOG, CNN PyTorch, ResNet18 avec transfert d'apprentissage)

Structure attendue du dataset (un sous-dossier par classe, son nom sert d'étiquette) :
    dataset_reel/
        normal/       normal_00001.png, ...
        pneumonie/    pneumonie_00001.png, ...

Installation :
    pip install numpy pillow scikit-learn scikit-image        # suffisant pour svm et rf
    pip install torch torchvision                              # pour pytorch et resnet

Utilisation :
    python classification_images_medicales.py --data dataset_reel --method svm
    python classification_images_medicales.py --data dataset_reel --method rf
    python classification_images_medicales.py --data dataset_reel --method pytorch --epochs 15
    python classification_images_medicales.py --data dataset_reel --method resnet --epochs 9
    python classification_images_medicales.py --data dataset_reel --method resnet --sans-preentrainement
    python classification_images_medicales.py --data dataset_reel --method svm --json resultats.json
    python classification_images_medicales.py --data dataset_reel --method svm --image dataset_reel/pneumonie/pneumonie_00001.png

Attention : code à but pédagogique, il ne constitue pas un outil de diagnostic.
"""
import argparse
import os

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

TAILLE = 128  # les images sont redimensionnées en 128 x 128
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------
# 1. Chargement et prétraitement
# ----------------------------------------------------------------------
def pretraiter(chemin):
    """Ouvre une image, la passe en niveaux de gris, la redimensionne, normalise à [0, 1]."""
    img = Image.open(chemin).convert("L").resize((TAILLE, TAILLE))
    return np.asarray(img, dtype="float32") / 255.0


def charger_dataset(dossier):
    classes = sorted(d for d in os.listdir(dossier) if os.path.isdir(os.path.join(dossier, d)))
    X, y = [], []
    for indice, classe in enumerate(classes):
        chemin_classe = os.path.join(dossier, classe)
        for fichier in os.listdir(chemin_classe):
            if fichier.lower().endswith(EXTENSIONS):
                X.append(pretraiter(os.path.join(chemin_classe, fichier)))
                y.append(indice)
    return np.array(X, dtype="float32"), np.array(y), classes


def poids_classes(y):
    """Poids inversement proportionnels à la fréquence (utile si classes déséquilibrées)."""
    classes = np.unique(y)
    poids = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(p) for c, p in zip(classes, poids)}


RESULTATS = {}  # métriques de chaque méthode, exportables avec --json


def evaluer(titre, y_vrai, y_pred, classes):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_vrai, y_pred, average="macro", zero_division=0)
    RESULTATS[titre] = {
        "accuracy": float(accuracy_score(y_vrai, y_pred)),
        "precision_macro": float(p), "rappel_macro": float(r), "f1_macro": float(f),
        "rapport": classification_report(y_vrai, y_pred, target_names=classes, output_dict=True, zero_division=0),
        "matrice": confusion_matrix(y_vrai, y_pred).tolist(),
    }
    print(f"\n===== {titre} =====")
    print(f"Exactitude (accuracy) : {accuracy_score(y_vrai, y_pred):.3f}")
    print(classification_report(y_vrai, y_pred, target_names=classes, zero_division=0))
    print("Matrice de confusion :")
    print(confusion_matrix(y_vrai, y_pred))


# ----------------------------------------------------------------------
# 2. Méthodes classiques : descripteurs HOG + SVM ou Random Forest
# ----------------------------------------------------------------------
def extraire_hog(images):
    from skimage.feature import hog

    return np.array([
        hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
        for img in images
    ])


def methode_classique(nom, Xtr, ytr, Xte, yte, classes):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    Ftr, Fte = extraire_hog(Xtr), extraire_hog(Xte)
    if nom == "svm":
        modele = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, class_weight="balanced", probability=True),
        )
    else:
        modele = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

    modele.fit(Ftr, ytr)
    evaluer("SVM + HOG" if nom == "svm" else "Random Forest + HOG", yte, modele.predict(Fte), classes)
    return lambda img: modele.predict_proba(extraire_hog(img[None]))[0]


# ----------------------------------------------------------------------
# 3. Réseau de neurones convolutif (CNN) avec PyTorch
# ----------------------------------------------------------------------
def methode_pytorch(Xtr, ytr, Xte, yte, classes, epochs):
    import torch
    import torch.nn as nn

    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    def bloc(entree, sortie):
        return nn.Sequential(nn.Conv2d(entree, sortie, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    modele = nn.Sequential(
        bloc(1, 32), bloc(32, 64), bloc(64, 128),
        nn.Flatten(), nn.Dropout(0.5),
        nn.Linear(128 * (TAILLE // 8) ** 2, 128), nn.ReLU(),
        nn.Linear(128, len(classes)),
    ).to(appareil)

    poids = poids_classes(ytr)
    perte = nn.CrossEntropyLoss(
        weight=torch.tensor([poids[i] for i in range(len(classes))], dtype=torch.float32).to(appareil)
    )
    optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)

    X_t = torch.tensor(Xtr[:, None])
    y_t = torch.tensor(ytr, dtype=torch.long)

    for epoque in range(epochs):
        modele.train()
        ordre = torch.randperm(len(X_t))
        total = 0.0
        for i in range(0, len(X_t), 32):
            idx = ordre[i:i + 32]
            xb, yb = X_t[idx].to(appareil), y_t[idx].to(appareil)
            optimiseur.zero_grad()
            l = perte(modele(xb), yb)
            l.backward()
            optimiseur.step()
            total += l.item() * len(idx)
        print(f"Époque {epoque + 1}/{epochs} - perte : {total / len(X_t):.4f}")

    modele.eval()

    def probas(images):
        sorties = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                lot = torch.tensor(images[i:i + 64][:, None]).to(appareil)
                sorties.append(torch.softmax(modele(lot), dim=1).cpu().numpy())
        return np.concatenate(sorties)

    evaluer("CNN PyTorch", yte, probas(Xte).argmax(axis=1), classes)
    return lambda img: probas(img[None])[0]


# ----------------------------------------------------------------------
# 4. Transfer learning : ResNet18 pré-entraîné sur ImageNet (PyTorch)
# ----------------------------------------------------------------------
def methode_resnet(Xtr, ytr, Xte, yte, classes, epochs, pretrained=True):
    import torch
    import torch.nn as nn
    from torchvision import models

    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    # pretrained=True : poids ImageNet (téléchargés au 1er lancement, connexion internet requise)
    # pretrained=False : ResNet18 initialisé au hasard (aucun transfert d'apprentissage)
    poids_init = models.ResNet18_Weights.DEFAULT if pretrained else None
    modele = models.resnet18(weights=poids_init)
    modele.fc = nn.Linear(modele.fc.in_features, len(classes))
    modele = modele.to(appareil)

    moyenne = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ecart = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preparer(images):
        """Niveaux de gris -> 3 canaux, puis normalisation ImageNet."""
        t = torch.tensor(images[:, None]).repeat(1, 3, 1, 1)
        return (t - moyenne) / ecart

    poids = poids_classes(ytr)
    perte = nn.CrossEntropyLoss(
        weight=torch.tensor([poids[i] for i in range(len(classes))], dtype=torch.float32).to(appareil)
    )
    X_t, y_t = preparer(Xtr), torch.tensor(ytr, dtype=torch.long)

    # Phase 1 : on entraîne seulement la dernière couche (le reste est gelé)
    # Phase 2 : on affine tout le réseau avec un petit taux d'apprentissage
    n1 = max(1, epochs // 3)
    if pretrained:
        phases = [("couche finale", 1e-3, n1), ("affinage complet", 1e-4, epochs - n1)]
    else:
        phases = [("entraînement complet", 1e-3, epochs)]

    for nom_phase, lr, n_ep in phases:
        for nom, p in modele.named_parameters():
            p.requires_grad = nom_phase != "couche finale" or nom.startswith("fc")
        optimiseur = torch.optim.Adam([p for p in modele.parameters() if p.requires_grad], lr=lr)
        for epoque in range(n_ep):
            modele.train()
            ordre = torch.randperm(len(X_t))
            total = 0.0
            for i in range(0, len(X_t), 32):
                idx = ordre[i:i + 32]
                xb, yb = X_t[idx].to(appareil), y_t[idx].to(appareil)
                optimiseur.zero_grad()
                l = perte(modele(xb), yb)
                l.backward()
                optimiseur.step()
                total += l.item() * len(idx)
            print(f"[{nom_phase}] époque {epoque + 1}/{n_ep} - perte : {total / len(X_t):.4f}")

    modele.eval()

    def probas(images):
        sorties = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                lot = preparer(images[i:i + 64]).to(appareil)
                sorties.append(torch.softmax(modele(lot), dim=1).cpu().numpy())
        return np.concatenate(sorties)

    evaluer("Transfer learning ResNet18" if pretrained else "ResNet18 sans pré-entraînement", yte, probas(Xte).argmax(axis=1), classes)
    return lambda img: probas(img[None])[0]


# ----------------------------------------------------------------------
# 5. Programme principal
# ----------------------------------------------------------------------
def main():
    parseur = argparse.ArgumentParser(description="Classification d'images médicales")
    parseur.add_argument("--data", default="dataset_reel",
                         help="dossier du dataset (un sous-dossier par classe)")
    parseur.add_argument("--method", default="svm", choices=["svm", "rf", "pytorch", "resnet", "all"])
    parseur.add_argument("--epochs", type=int, default=20)
    parseur.add_argument("--image", help="image à classifier après l'entraînement")
    parseur.add_argument("--sans-preentrainement", action="store_true",
                         help="ResNet18 sans poids ImageNet (utile hors connexion)")
    parseur.add_argument("--json", help="fichier où enregistrer les résultats (JSON)")
    args = parseur.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Dossier '{args.data}' introuvable. Voir telecharger_dataset_reel.py pour obtenir des images.")

    X, y, classes = charger_dataset(args.data)
    print(f"{len(X)} images chargées, classes : {classes}")
    for i, c in enumerate(classes):
        print(f"  - {c} : {(y == i).sum()} images")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    methodes = ["svm", "rf", "pytorch", "resnet"] if args.method == "all" else [args.method]
    import json, time
    temps = {}
    for m in methodes:
        debut = time.time()
        if m in ("svm", "rf"):
            predire = methode_classique(m, Xtr, ytr, Xte, yte, classes)
        elif m == "pytorch":
            predire = methode_pytorch(Xtr, ytr, Xte, yte, classes, args.epochs)
        else:
            predire = methode_resnet(Xtr, ytr, Xte, yte, classes, args.epochs, not args.sans_preentrainement)

        temps[m] = time.time() - debut
        print(f"Durée ({m}) : {temps[m]:.1f} s")
        if args.image:
            proba = predire(pretraiter(args.image))
            k = int(np.argmax(proba))
            print(f"-> Prédiction ({m}) pour {args.image} : {classes[k]} ({100 * proba[k]:.1f} %)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"classes": classes, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                       "resultats": RESULTATS, "temps_s": temps}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
