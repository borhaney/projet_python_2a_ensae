# Saint-Ouen en mutation : recompositions socio-spatiales et dynamiques urbaines depuis les années 2010

Projet Python pour la data science (ENSAE, 2A, 2026) réalisé par Borhane Yahiaoui et Pauline Aurignac.

## Présentation

Commune limitrophe du nord de Paris, Saint-Ouen-sur-Seine s'est profondément transformée depuis les années 2010. Ce projet examine, à l'aide de données publiques et à l'échelle des IRIS (Îlots Regroupés pour l'Information Statistique), plusieurs dimensions de cette évolution : démographie, niveau de vie, composition sociale, structure des ménages et marché immobilier.

Problématique : dans quelle mesure les transformations récentes de Saint-Ouen s'accompagnent-elles d'une recomposition socio-spatiale, et comment celle-ci se différencie-t-elle selon les quartiers ?

L'approche est descriptive : elle mesure des évolutions et des corrélations spatiales et temporelles, sans prétendre identifier des relations de cause à effet.

## Contenu du dépôt

```
.
├── main.ipynb          # Notebook principal : analyse, cartes et commentaires
├── utils.py            # Fonctions de chargement, nettoyage, calcul et tracé
├── requirements.txt    # Dépendances Python
├── data/
│   ├── raw/            # Données brutes à placer ici (voir ci-dessous)
│   └── processed/      # Données nettoyées générées par le notebook
└── README.md
```

## Plan du notebook

- Introduction et problématique
- I. Création et exploration des jeux de données : données géographiques (IRIS), données immobilières (DVF), données Filosofi, données couples-familles-ménages, données d'activité des résidents
- II. Évolution démographique : pyramides des âges 2012 et 2023
- III. Transformations socio-économiques : niveau de vie (2018-2021), part des cadres et mixité sociale (2011-2022), structure des ménages (2011-2022)
- IV. Logements et transactions : prix au m² selon la proximité des stations de métro, carte des prix par IRIS
- Conclusion

## Données utilisées

| Source | Contenu | Millésimes |
|---|---|---|
| Insee, contours IRIS | Géométrie des IRIS (`IDF_iris.geojson`) | n.c. |
| DVF (data.gouv.fr) | Transactions immobilières | 2018 à 2025 |
| Insee, Filosofi | Niveau de vie médian par IRIS | 2018, 2021 |
| Insee, base infra-communale couples-familles-ménages | Structure des ménages par IRIS | 2011, 2022 |
| Insee, base infra-communale activité des résidents | Catégories socioprofessionnelles par IRIS | 2011, 2022 |
| Insee, API Melodi | Population par sexe et âge de la commune | 2012, 2023 |

Les fichiers bruts ne sont pas versionnés. Il faut les télécharger et les placer dans `data/raw/` avec les noms attendus par le notebook :

- `IDF_iris.geojson`
- les fichiers DVF : fichiers `.csv` limités à Saint-Ouen (par exemple `93070.2021.csv`) et fichiers nationaux `.parquet` pour les années plus anciennes (par exemple `dvf-2018.parquet`)
- `BASE_TD_FILO_DISP_IRIS_2018.xlsx` et `BASE_TD_FILO_IRIS_2021_DISP.xlsx`
- `base-ic-couples-familles-menages-2011.xls` et `base-ic-couples-familles-menages-2022.xlsx`
- `base-ic-activite-residents-2011.xls` et `base-ic-activite-residents-2022.xlsx`

Certaines données (par exemple la population communale via l'API Melodi) sont récupérées en ligne : une connexion internet est nécessaire pour exécuter ces cellules.

## Installation et exécution

1. Cloner le dépôt et se placer à sa racine.
2. Créer et activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Placer les fichiers de données dans `data/raw/`.
5. Ouvrir `main.ipynb` et exécuter les cellules dans l'ordre. Le notebook doit être lancé depuis la racine du projet, car les chemins vers les données sont relatifs.

Remarque : la lecture des fichiers `.xls` nécessite le paquet `python-calamine`, et celle des fichiers `.xlsx` le paquet `openpyxl`. Vérifier qu'ils figurent dans `requirements.txt`.

## Principaux choix méthodologiques

- Les codes IRIS sont normalisés sur 9 caractères pour permettre les jointures entre sources.
- Les valeurs non numériques des tables Insee deviennent des valeurs manquantes, sans imputation.
- Pour les données DVF, une observation correspond à une mutation. Seules les ventes comportant au moins un appartement ou une maison sont conservées, avec des seuils sur la surface, la valeur foncière et le prix au m². Les seuils sont des choix de nettoyage documentés dans le notebook.
- Les évolutions sont calculées en points de pourcentage pour les parts, et en variation relative pour le niveau de vie.
- Les cartes d'évolution utilisent des palettes divergentes centrées sur zéro.

## Principaux résultats

- Entre 2012 et 2023, la population augmente surtout dans les classes d'âge actif.
- Le niveau de vie médian progresse dans tous les IRIS entre 2018 et 2021, plus fortement dans certains quartiers.
- La part des cadres augmente dans la quasi-totalité des IRIS entre 2011 et 2022, tandis que la mixité sociale évolue de façon contrastée selon les quartiers.
- La structure des ménages évolue différemment au nord (progression des familles monoparentales et nombreuses) et au sud (progression des ménages d'une personne et des couples sans enfant).
- Les prix au m² sont proches entre zone proche et zone éloignée des stations en 2018, avec un écart modéré à partir de 2021.

## Limites

- L'analyse est descriptive et ne permet pas de conclure à des effets causaux.
- Les sources couvrent des périodes différentes, ce qui limite la comparaison directe entre cartes.
- Les IRIS sont de petite taille et les données du recensement reposent sur un sondage.
- Le prix au m² est un indicateur approximatif calculé à l'échelle de la mutation.
- La distance aux stations est calculée depuis le centroïde des IRIS, pas depuis chaque logement vendu, dans l'analyse par IRIS.
