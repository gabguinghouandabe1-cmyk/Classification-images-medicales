# Classification d'images médicales avec Python

Programme Python qui classe des radiographies thoraciques en deux catégories (*normal* / *pneumonie*)
et compare **quatre méthodes** :

| Option `--method` | Méthode |
|---|---|
| `svm` | SVM + descripteurs HOG |
| `rf` | Random Forest + descripteurs HOG |
| `pytorch` | CNN (PyTorch) |
| `resnet` | ResNet18 (transfert d'apprentissage, poids ImageNet) |

Pour chaque méthode, le programme affiche l'exactitude, la précision, le rappel, le F1 et la matrice de confusion.

> **Avertissement** : projet pédagogique (Master MIB, Université de Dschang). Ce programme n'est ni validé
> ni certifié et ne doit pas servir à établir un diagnostic.

## Exemples d'images

Radiographies du jeu PneumoniaMNIST (CC BY 4.0).

![Exemples normal](exemples_normal.png)
![Exemples pneumonie](exemples_pneumonie.png)

## Installation

```bash
pip install numpy pillow scikit-learn scikit-image torch torchvision medmnist
```

## Utilisation

```bash
python telecharger_dataset_reel.py --max 1000 --size 64     # télécharge PneumoniaMNIST dans dataset_reel/
python classification_images_medicales.py --data dataset_reel --method svm
python classification_images_medicales.py --data dataset_reel --method rf
python classification_images_medicales.py --data dataset_reel --method pytorch --epochs 15
python classification_images_medicales.py --data dataset_reel --method resnet --epochs 9
```

Classer une image précise :

```bash
python classification_images_medicales.py --data dataset_reel --method svm --image dataset_reel/pneumonie/pneumonie_00001.png
```

Ses propres images : un dossier avec **un sous-dossier par classe** (le nom du sous-dossier sert d'étiquette).

## Options

| Option | Rôle |
|---|---|
| `--data` | dossier d'images (défaut : `dataset_reel`) |
| `--method` | `svm` (défaut), `rf`, `pytorch`, `resnet`, `all` |
| `--epochs` | nombre d'époques pour les réseaux de neurones (défaut 20) |
| `--image` | classer une image précise après l'entraînement |
| `--sans-preentrainement` | ResNet18 sans poids ImageNet |
| `--json` | enregistrer les métriques dans un fichier JSON |

## Résultats

2 000 images (1 000 par classe), 1 600 en entraînement, 400 en test, images 64 × 64 agrandies à 128 × 128.
Classe positive : pneumonie.

| Méthode | Accuracy | Précision | Rappel | F1 | Durée (s) |
|---|---|---|---|---|---|
| SVM + HOG | 97,0 % | 99,5 % | 94,5 % | 96,9 % | 15 |
| Random Forest + HOG | 93,5 % | 95,3 % | 91,5 % | 93,4 % | 30 |
| CNN PyTorch (15 époques) | 97,0 % | 98,0 % | 96,0 % | 97,0 % | 679 |
| ResNet18 ImageNet (9 époques) | 98,0 % | 99,5 % | 96,5 % | 98,0 % | 846 |

Matrices de confusion (VP : pneumonies détectées ; FN : pneumonies manquées ; FP : fausses alarmes ; VN : normales reconnues) :

| Méthode | VP | FN | FP | VN |
|---|---|---|---|---|
| SVM + HOG | 189 | 11 | 1 | 199 |
| Random Forest + HOG | 183 | 17 | 9 | 191 |
| CNN PyTorch | 192 | 8 | 4 | 196 |
| ResNet18 | 193 | 7 | 1 | 199 |

Limites : découpage par image et non par patient, basse résolution, une seule exécution par méthode.

## Fichiers

- `classification_images_medicales.py` : le programme principal
- `telecharger_dataset_reel.py` : téléchargement des images (PneumoniaMNIST)

## Références

- Kermany, D. S. et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5).
- Yang, J. et al. (2023). MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10.
