"""
Télécharge de VRAIES radiographies thoraciques (jeu PneumoniaMNIST de MedMNIST v3)
et les range dans le format attendu par classification_images_medicales.py :

    dataset_reel/
        normal/       *.png
        pneumonie/    *.png

Source : radiographies pédiatriques de Kermany et al. (2018), reprises par
MedMNIST (Yang et al., 2023), licence CC BY 4.0. Environ 5 800 images.

Installation (connexion internet nécessaire) :
    pip install medmnist numpy pillow

Utilisation :
    python telecharger_dataset_reel.py                  # images 128 x 128 -> dossier dataset_reel
    python telecharger_dataset_reel.py --max 500        # au plus 500 images par classe (plus rapide)
    python classification_images_medicales.py --data dataset_reel --method rf

Alternative manuelle : télécharger « Chest X-Ray Images (Pneumonia) » sur Kaggle, dézipper,
puis lancer :  python classification_images_medicales.py --data chest_xray/train
"""
import argparse
import os

import numpy as np
from PIL import Image

NOMS_FR = {"normal": "normal", "pneumonia": "pneumonie"}


def sauvegarder(images, etiquettes, noms_classes, racine, compteurs, maximum=None):
    """Enregistre chaque image en PNG dans racine/<classe>/. Retourne les compteurs mis à jour."""
    for img, lab in zip(images, np.asarray(etiquettes).reshape(-1)):
        nom = noms_classes[int(lab)].lower()
        nom = NOMS_FR.get(nom, nom)
        if maximum is not None and compteurs.get(nom, 0) >= maximum:
            continue
        dossier = os.path.join(racine, nom)
        os.makedirs(dossier, exist_ok=True)
        compteurs[nom] = compteurs.get(nom, 0) + 1
        Image.fromarray(np.asarray(img, dtype="uint8")).save(
            os.path.join(dossier, f"{nom}_{compteurs[nom]:05d}.png")
        )
    return compteurs


def main():
    parseur = argparse.ArgumentParser(description="Télécharge PneumoniaMNIST (vraies radiographies)")
    parseur.add_argument("--out", default="dataset_reel", help="dossier de sortie")
    parseur.add_argument("--size", type=int, default=128, choices=[28, 64, 128, 224])
    parseur.add_argument("--max", type=int, default=None, help="nombre max d'images par classe")
    args = parseur.parse_args()

    try:
        from medmnist import PneumoniaMNIST, INFO
    except ImportError:
        raise SystemExit("Installez d'abord la bibliothèque :  pip install medmnist")

    noms = {int(k): v for k, v in INFO["pneumoniamnist"]["label"].items()}
    compteurs = {}
    for split in ("train", "val", "test"):
        print(f"Téléchargement du sous-ensemble '{split}'...")
        ds = PneumoniaMNIST(split=split, download=True, size=args.size)
        sauvegarder(ds.imgs, ds.labels, noms, args.out, compteurs, args.max)

    print(f"\nTerminé : images enregistrées dans '{args.out}'")
    for nom, n in sorted(compteurs.items()):
        print(f"  - {nom} : {n} images")
    print(f"\nLancez maintenant :\n  python classification_images_medicales.py --data {args.out} --method rf")


if __name__ == "__main__":
    main()
