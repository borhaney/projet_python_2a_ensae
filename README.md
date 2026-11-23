# Saint-Ouen en mutation

Projet Python pour la data science — Borhane Yahiaoui et Pauline Aurignac (2026).

Le projet étudie les transformations démographiques, socio-économiques, résidentielles et immobilières de Saint-Ouen-sur-Seine depuis les années 2010, ainsi que les différences entre quartiers (IRIS). Sa problématique est : « Dans quelle mesure les transformations récentes de Saint-Ouen s’accompagnent-elles d’une recomposition socio-spatiale, et comment celle-ci se différencie-t-elle selon les quartiers ? » L’approche présentée dans le notebook est descriptive et ne prétend pas établir de causalité.

## Livrable de référence

Le document à consulter pour l’évaluation est **[`main.ipynb`](main.ipynb)**. Les notebooks [`notebook_borhane.ipynb`](notebook_borhane.ipynb) et [`notebook_pauline.ipynb`](notebook_pauline.ipynb) ne font pas partie du livrable à évaluer et ne sont pas nécessaires pour suivre le projet.

## Exécution

Depuis la racine du dépôt, préparer un environnement Python, installer les dépendances, puis ouvrir `main.ipynb` dans Jupyter ou un éditeur compatible et exécuter ses cellules dans l’ordre :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pyarrow
```

`pyarrow` est importé par le notebook pour lire les fichiers Parquet, mais n’est pas déclaré dans `requirements.txt` ; il est donc installé séparément ci-dessus.

### Données locales requises

Le notebook attend les données brutes dans `data/raw/`, notamment le fond géographique `IDF_iris.geojson`, ainsi que les fichiers DVF. Il ouvre explicitement `93070.2021.csv` et `dvf-2018.parquet` et recherche aussi des fichiers annuels DVF (Parquet ou CSV `93070.<année>.csv`, pour les années 2017 à 2025). Après le nettoyage et la jointure spatiale, `utils.py` écrit le fichier dérivé `data/processed/data_immo.parquet`.

**Ces données brutes ne sont pas présentes dans le dépôt courant** : le dossier `data/` n’y figure pas. Il faut donc obtenir les fichiers sources et les placer dans l’arborescence attendue avant d’exécuter les cellules qui les chargent. Le dépôt ne fournit pas de procédure automatique de téléchargement de ces données.

`utils.py` contient notamment les fonctions utilisées dans le notebook pour charger les IRIS et les DVF, nettoyer les transactions et les rattacher aux IRIS. Son chargeur de la ligne 14 est indépendant et n’est pas utilisé dans l’analyse présentée.
