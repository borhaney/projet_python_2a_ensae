# ==============================================================================
# Imports et constantes
# ==============================================================================

from pathlib import Path
import re
import unicodedata
from urllib.parse import urlencode

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

# Colonne IRIS utilisée dans les jointures et agrégations
COL_IRIS = "code_iris"

DOSSIER_PROJET = Path(__file__).resolve().parent

# Paramètres API Sitadel (construction de logements)
DEPARTEMENT = "93"
ANNEES = range(2011, 2023)
DATASET_SLUG = (
    "logements-autorises-et-commences-nombre-et-surfaces-"
    "series-mensuelles-donnees-estimees-1"
)
DATAGOUV_API_URL = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_SLUG}/"
TABULAR_API_BASE = "https://tabular-api.data.gouv.fr/api/resources"

# Paramètres API Melodi (Insee, population)
URL_MELODI = "https://api.insee.fr/melodi/data"
TABLE_POPULATION_MELODI = "DS_RP_POPULATION_PRINC"
ANNEES_POPULATION_MELODI = (2012, 2017, 2023)


# ==============================================================================
# 1. Périmètre IRIS
# ==============================================================================


def charger_zone_etude(code_insee="93070"):
    """Charge les polygones IRIS de la commune ciblée.

    Returns
    -------
    GeoDataFrame
        Un polygone par IRIS de la commune, avec son identifiant normalisé
        dans la colonne ``code_iris``.
    """
    fichier_iris = DOSSIER_PROJET / "data" / "raw" / "IDF_iris.geojson"
    if not fichier_iris.is_file():
        raise FileNotFoundError(f"Fichier IRIS introuvable : {fichier_iris}")

    gdf_iris = gpd.read_file(fichier_iris)
    colonnes_requises = {"insee_com", COL_IRIS}
    colonnes_absentes = colonnes_requises - set(gdf_iris.columns)
    if colonnes_absentes:
        raise ValueError(
            "Colonnes obligatoires absentes du fond IRIS : "
            f"{sorted(colonnes_absentes)}. "
            f"Colonnes disponibles : {list(gdf_iris.columns)}"
        )

    code_insee = str(code_insee).strip().removesuffix(".0").zfill(5)
    codes = (
        gdf_iris["insee_com"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    zone_etude = gdf_iris.loc[codes == code_insee].copy()
    if zone_etude.empty:
        raise ValueError(
            f"Aucun IRIS trouvé pour la commune {code_insee} dans 'insee_com'."
        )
    if zone_etude.crs is None:
        raise ValueError("Le fichier IRIS ne précise pas son système de coordonnées.")

    zone_etude[COL_IRIS] = zone_etude[COL_IRIS].astype("string")
    return zone_etude


def charger_iris_api(commune_prefix: str = "93070", session=None) -> gpd.GeoDataFrame:
    """Récupère les contours IRIS via le WFS IGN en priorité, avec repli sur
    le fichier local (iris.gpkg puis IDF_iris.geojson)."""
    if session is None:
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; EnsaeGeoProject/1.0)"}
        )

    candidats_bbox = [
        "2.310,48.895,2.360,48.925,urn:ogc:def:crs:EPSG::4326",
        "48.895,2.310,48.925,2.360,urn:ogc:def:crs:EPSG::4326",
    ]

    # 1. Tentative via l'API WFS IGN
    for bbox in candidats_bbox:
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAME": "STATISTICALUNITS.IRIS:contours_iris",
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": "EPSG:4326",
            "COUNT": "200",
            "BBOX": bbox,
        }
        url = f"https://data.geopf.fr/wfs/ows?{urlencode(params)}"
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            gdf = gpd.read_file(resp.text)
            if not gdf.empty:
                gdf_norm = _normaliser_colonne_iris(gdf)
                st_ouen = gdf_norm[
                    gdf_norm[COL_IRIS]
                    .astype(str)
                    .str.strip()
                    .str.startswith(commune_prefix)
                ].copy()
                if not st_ouen.empty:
                    st_ouen[COL_IRIS] = (
                        st_ouen[COL_IRIS].astype(str).str.strip().str.zfill(9)
                    )
                    print("-> Contours IRIS chargés via WFS IGN.")
                    return st_ouen
        except Exception:
            continue

    # 2. Repli automatique sur un fichier local
    print("API WFS IGN injoignable, bascule sur un fichier local...")
    candidats_locaux = (
        DOSSIER_PROJET / "iris.gpkg",
        DOSSIER_PROJET / "data" / "raw" / "IDF_iris.geojson",
    )
    fichier_iris = next((c for c in candidats_locaux if c.is_file()), None)
    if fichier_iris is None:
        raise FileNotFoundError(
            "Échec de l'API et fichiers locaux introuvables : "
            + ", ".join(str(c) for c in candidats_locaux)
        )

    gdf_iris = gpd.read_file(fichier_iris)

    # Détection automatique de la colonne de commune (insensible à la casse)
    noms_possibles = {
        "insee_com",
        "INSEE_COM",
        "code_insee",
        "CODE_INSEE",
        "depcom",
        "DEPCOM",
    }
    col_com = next((c for c in noms_possibles if c in gdf_iris.columns), None)
    if col_com is None:
        raise ValueError(
            f"Aucune colonne de code commune trouvée dans {fichier_iris.name}. "
            f"Colonnes disponibles : {list(gdf_iris.columns)}"
        )

    code_insee = str(commune_prefix).strip().removesuffix(".0").zfill(5)
    codes = (
        gdf_iris[col_com]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    zone_etude = gdf_iris.loc[codes == code_insee].copy()
    if zone_etude.empty:
        raise ValueError(
            f"Aucun IRIS trouvé pour la commune {code_insee} "
            f"dans {fichier_iris.name}."
        )
    if zone_etude.crs is None:
        raise ValueError(
            f"Le fichier {fichier_iris.name} ne précise pas son système de "
            "coordonnées."
        )

    zone_etude[COL_IRIS] = (
        zone_etude[COL_IRIS].astype("string").str.strip().str.zfill(9)
    )
    print(f"-> Contours IRIS chargés depuis le fichier local {fichier_iris.name}.")
    return zone_etude


def _normaliser_colonne_iris(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalise l'identifiant IRIS d'un GeoDataFrame WFS dans ``code_iris``."""
    col_iris_src = next(
        (
            c
            for c in ("iris", "CODE_IRIS", "code_iris", "IRIS", "insee_iris")
            if c in gdf.columns
        ),
        None,
    )
    if col_iris_src is None:
        raise ValueError(
            "Aucune colonne identifiant IRIS trouvée. "
            f"Colonnes disponibles : {list(gdf.columns)}"
        )
    if col_iris_src != COL_IRIS:
        gdf = gdf.rename(columns={col_iris_src: COL_IRIS})
    gdf[COL_IRIS] = gdf[COL_IRIS].astype(str)
    return gdf


# ==============================================================================
# 2. Population communale (API Melodi, Insee)
# ==============================================================================


def _charger_observations_melodi(tableau, code_geo, annees):
    """Télécharge et aplatit des observations de l'API Melodi de l'Insee."""
    observations = []
    for annee in annees:
        try:
            response = requests.get(
                f"{URL_MELODI}/{tableau}",
                params={"GEO": code_geo, "TIME_PERIOD": str(annee)},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Échec du chargement Melodi pour {code_geo}, année {annee}."
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Réponse JSON Melodi invalide pour {code_geo}, année {annee}."
            ) from exc

        lignes = payload.get("observations")
        if not isinstance(lignes, list):
            raise ValueError(
                "La réponse Melodi ne contient pas de liste 'observations'."
            )
        if not lignes:
            raise ValueError(
                f"Aucune observation Melodi pour {code_geo}, année {annee}."
            )

        for ligne in lignes:
            dimensions = ligne.get("dimensions")
            if not isinstance(dimensions, dict):
                raise ValueError("Une observation Melodi n'a pas de dimensions.")
            valeur = (
                ligne.get("measures", {})
                .get("OBS_VALUE_NIVEAU", {})
                .get("value")
            )
            observations.append(
                {
                    "code_geo": dimensions.get("GEO"),
                    "sexe": dimensions.get("SEX"),
                    "annee": int(dimensions.get("TIME_PERIOD", annee)),
                    "indicateur": dimensions.get("RP_MEASURE"),
                    "age": dimensions.get("AGE"),
                    "population": valeur,
                    "statut": ligne.get("attributes", {}).get("OBS_STATUS"),
                }
            )

    resultat = pd.DataFrame(observations)
    if resultat["population"].isna().any():
        raise ValueError("La réponse Melodi contient des populations manquantes.")
    resultat["population"] = pd.to_numeric(resultat["population"], errors="raise")
    return resultat


def charger_population_commune(code_insee="93070", annees=None):
    """Charge la population communale par sexe et groupes d'âge via Melodi.

    L'API diffuse actuellement les millésimes 2012, 2017 et 2023 pour ce
    tableau. Ce sont des années de référence du recensement, pas des données
    pour 2011 ou l'année courante.
    """
    code_insee = str(code_insee).strip().removesuffix(".0")
    if not code_insee.isdigit() or len(code_insee) > 5:
        raise ValueError(f"Code INSEE communal invalide : {code_insee!r}.")
    code_insee = code_insee.zfill(5)

    if annees is None:
        annees = ANNEES_POPULATION_MELODI
    elif isinstance(annees, (int, str)):
        annees = (int(annees),)
    else:
        annees = tuple(int(annee) for annee in annees)
    if not annees:
        raise ValueError("La liste des années Melodi ne peut pas être vide.")
    if len(set(annees)) != len(annees):
        raise ValueError("Les années Melodi doivent être uniques.")

    return _charger_observations_melodi(
        TABLE_POPULATION_MELODI,
        code_geo=f"COM-{code_insee}",
        annees=annees,
    )


def preparer_pyramide_ages(population_commune, annee=None):
    """Prépare les effectifs par âge et sexe pour le graphique de pyramide.

    Par défaut, retourne 2017 et 2023 si ces millésimes sont présents;
    sinon, les millésimes disponibles le plus ancien et le plus récent.
    """
    colonnes_requises = {"annee", "sexe", "age", "population", "indicateur"}
    colonnes_absentes = colonnes_requises - set(population_commune.columns)
    if colonnes_absentes:
        raise ValueError(
            "Colonnes absentes des données de population : "
            f"{sorted(colonnes_absentes)}."
        )

    annees_colonne = pd.to_numeric(population_commune["annee"], errors="coerce")
    annees_disponibles = sorted(annees_colonne.dropna().astype(int).unique())
    if annee is None:
        if not annees_disponibles:
            raise ValueError("Aucune année valide dans les données de population.")
        if 2017 in annees_disponibles and 2023 in annees_disponibles:
            annees_a_conserver = [2017, 2023]
        else:
            annees_a_conserver = sorted(
                {annees_disponibles[0], annees_disponibles[-1]}
            )
        if len(annees_a_conserver) < 2:
            raise ValueError(
                "Il faut au moins deux millésimes pour comparer les pyramides."
            )
    elif isinstance(annee, (int, str)):
        annees_a_conserver = [int(annee)]
    else:
        annees_a_conserver = sorted({int(millesime) for millesime in annee})
        if not annees_a_conserver:
            raise ValueError("La liste des années ne peut pas être vide.")

    groupes_age = {
        "Y_LT15": "0–14 ans",
        "Y15T24": "15–24 ans",
        "Y25T39": "25–39 ans",
        "Y40T54": "40–54 ans",
        "Y55T64": "55–64 ans",
        "Y65T79": "65–79 ans",
        "Y_GE80": "80 ans ou plus",
    }
    donnees = population_commune.loc[
        annees_colonne.isin(annees_a_conserver)
        & population_commune["indicateur"].eq("POP")
        & population_commune["sexe"].isin(["F", "M"])
        & population_commune["age"].isin(groupes_age)
    ]
    if donnees.empty:
        raise ValueError(
            "Aucune donnée de pyramide pour les années "
            f"{annees_a_conserver}."
        )

    pyramide = donnees.pivot(
        index=["annee", "age"],
        columns="sexe",
        values="population",
    ).reset_index()
    pyramide = pyramide.loc[
        pyramide["annee"].isin(annees_a_conserver)
    ].copy()
    if not {"F", "M"}.issubset(pyramide.columns):
        raise ValueError("Les effectifs femmes/hommes sont incomplets.")
    if pyramide[["F", "M"]].isna().any().any():
        raise ValueError("Les effectifs femmes/hommes sont incomplets.")

    pyramide["annee"] = pyramide["annee"].astype(int)
    pyramide["classe_age"] = pyramide["age"].map(groupes_age)
    pyramide = pyramide.sort_values(
        ["annee", "age"],
        key=lambda colonne: (
            colonne.map({code: ordre for ordre, code in enumerate(groupes_age)})
            if colonne.name == "age"
            else colonne
        ),
    )
    pyramide = pyramide.rename(columns={"F": "femmes", "M": "hommes"})
    pyramide["tranche_age"] = pyramide["classe_age"]
    return pyramide[
        ["annee", "age", "classe_age", "tranche_age", "femmes", "hommes"]
    ].reset_index(drop=True)


def calculer_evolution_population_commune(
    population_commune,
    annee_debut=2012,
    annee_fin=2023,
):
    """Calcule l'évolution de la population totale entre deux millésimes."""
    colonnes_requises = {"annee", "sexe", "age", "population", "indicateur"}
    colonnes_absentes = colonnes_requises - set(population_commune.columns)
    if colonnes_absentes:
        raise ValueError(
            "Colonnes absentes des données de population : "
            f"{sorted(colonnes_absentes)}."
        )

    annees = pd.to_numeric(population_commune["annee"], errors="coerce")
    totaux = population_commune.loc[
        population_commune["indicateur"].eq("POP")
        & population_commune["sexe"].eq("_T")
        & population_commune["age"].eq("_T")
    ].assign(annee=annees)
    totaux = totaux.set_index("annee")["population"]

    valeurs = {}
    for annee in (int(annee_debut), int(annee_fin)):
        if annee not in totaux.index:
            raise ValueError(f"Population totale absente pour l'année {annee}.")
        selection = totaux.loc[annee]
        if isinstance(selection, pd.Series):
            if len(selection) != 1:
                raise ValueError(f"Population totale ambiguë pour l'année {annee}.")
            selection = selection.iloc[0]
        valeurs[annee] = float(selection)

    debut, fin = valeurs[int(annee_debut)], valeurs[int(annee_fin)]
    variation = fin - debut
    variation_pct = pd.NA if debut == 0 else variation / debut * 100
    return pd.DataFrame(
        [
            {
                "annee_debut": int(annee_debut),
                "annee_fin": int(annee_fin),
                "population_debut": debut,
                "population_fin": fin,
                "variation_absolue": variation,
                "variation_pourcent": variation_pct,
            }
        ]
    )


# ==============================================================================
# 3. Réseau de transport (Ligne 14 & gares)
# ==============================================================================


def _normaliser_identifiant_ligne(valeur):
    """Normalise un nom/code de ligne pour une comparaison exacte."""
    texte = unicodedata.normalize("NFKD", str(valeur))
    texte = texte.encode("ascii", "ignore").decode("ascii").upper()
    return re.sub(r"[^A-Z0-9]", "", texte)


def charger_ligne_14(
    zone_etude=None,
    colonne_ligne=None,
    valeur_ligne="14",
):
    """Charge et filtre le tracé de la ligne 14, sans charger les gares.

    Si le GeoJSON utilise un schéma différent, préciser ``colonne_ligne`` et
    ``valeur_ligne`` après avoir inspecté ses propriétés.
    """
    chemin = (
        DOSSIER_PROJET
        / "data"
        / "raw"
        / "traces-des-lignes-de-transport-en-commun-idfm-2.geojson"
    )
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier réseau introuvable : {chemin}")

    reseau = gpd.read_file(chemin)
    if reseau.empty:
        raise ValueError(f"Le fichier réseau est vide : {chemin}")
    if reseau.crs is None:
        raise ValueError(f"Le fichier réseau ne précise pas son CRS : {chemin}")

    colonnes_candidates = {
        "LIGNE",
        "NOMLIGNE",
        "CODELIGNE",
        "IDLIGNE",
        "NUMERO",
        "NUMEROLIGNE",
        "LINE",
        "LINENAME",
        "SHORTNAME",
        "LONGNAME",
        "ROUTESHORTNAME",
        "ROUTELONGNAME",
        "NOM",
        "NAME",
    }
    if colonne_ligne is not None:
        if colonne_ligne not in reseau.columns:
            raise ValueError(
                f"Colonne de ligne {colonne_ligne!r} absente. "
                f"Colonnes disponibles : {list(reseau.columns)}"
            )
        colonnes_a_tester = [colonne_ligne]
    else:
        colonnes_a_tester = [
            col
            for col in reseau.columns
            if _normaliser_identifiant_ligne(col) in colonnes_candidates
        ]

    valeurs_cibles = (
        {"14", "M14", "L14", "METRO14", "LIGNE14", "METROLIGNE14"}
        if str(valeur_ligne).strip() == "14"
        else {_normaliser_identifiant_ligne(valeur_ligne)}
    )
    masques = []
    for col in colonnes_a_tester:
        valeurs = reseau[col].astype("string").map(_normaliser_identifiant_ligne)
        masque = valeurs.isin(valeurs_cibles)
        if masque.any():
            masques.append(masque)

    if not masques:
        raise ValueError(
            "Impossible d'identifier la ligne demandée dans les propriétés "
            f"du GeoJSON. Colonnes disponibles : {list(reseau.columns)}. "
            "Précise colonne_ligne et valeur_ligne selon le schéma du fichier."
        )

    masque_ligne = masques[0]
    for masque in masques[1:]:
        masque_ligne = masque_ligne | masque
    ligne = reseau.loc[masque_ligne].copy()

    if zone_etude is not None:
        if zone_etude.crs is None:
            raise ValueError("La zone d'étude ne précise pas son CRS.")
        if ligne.crs != zone_etude.crs:
            ligne = ligne.to_crs(zone_etude.crs)

    return ligne


def charger_stations_proches(zone_etude, chemin=None, marge=0.05):
    """Charge les gares proches de la zone d'étude à partir du GeoJSON local."""
    if not isinstance(zone_etude, gpd.GeoDataFrame):
        raise TypeError("zone_etude doit être un GeoDataFrame.")
    if zone_etude.empty:
        raise ValueError("La zone d'étude est vide.")
    if zone_etude.crs is None:
        raise ValueError("La zone d'étude ne précise pas son CRS.")

    try:
        marge = float(marge)
    except (TypeError, ValueError) as exc:
        raise ValueError("marge doit être un nombre positif ou nul.") from exc
    if not pd.notna(marge) or marge < 0:
        raise ValueError("marge doit être un nombre positif ou nul.")

    if chemin is None:
        chemin = (
            DOSSIER_PROJET
            / "data"
            / "raw"
            / "emplacement-des-gares-idf.geojson"
        )
    else:
        chemin = Path(chemin)
        if not chemin.is_absolute():
            chemin = DOSSIER_PROJET / chemin
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier de gares introuvable : {chemin}")

    stations = gpd.read_file(chemin)
    if stations.empty:
        raise ValueError(f"Le fichier de gares est vide : {chemin}")
    if stations.crs is None:
        raise ValueError(f"Le fichier de gares ne précise pas son CRS : {chemin}")
    stations = stations.to_crs(zone_etude.crs)

    minx, miny, maxx, maxy = zone_etude.total_bounds
    marge_x = (maxx - minx) * marge
    marge_y = (maxy - miny) * marge
    stations_proches = stations.cx[
        minx - marge_x : maxx + marge_x,
        miny - marge_y : maxy + marge_y,
    ].copy()
    if stations_proches.empty:
        raise ValueError("Aucune gare trouvée autour du périmètre d'étude.")
    return stations_proches


def calculer_zones_tampons(
    zone_etude,
    stations,
    dist_traitee=500,
    dist_etendue=800,
):
    """Calcule la distance aux gares, les buffers et la classification
    Traité / Contrôle de chaque IRIS.

    Returns
    -------
    tuple
        (IRIS avec 'dist_station_m' et 'groupe',
         GeoDataFrame buffer traité, GeoDataFrame buffer étendu)
    """
    crs_orig = zone_etude.crs

    # 1. Reprojection en Lambert-93 (EPSG:2154) pour des distances métriques
    zone_2154 = zone_etude.to_crs("EPSG:2154")
    stations_2154 = stations.to_crs("EPSG:2154")

    # 2. Géométries des buffers unifiés
    union_stations = unary_union(stations_2154.geometry)
    gdf_buf_traite = gpd.GeoDataFrame(
        geometry=[union_stations.buffer(dist_traitee)],
        crs="EPSG:2154",
    )
    gdf_buf_etendu = gpd.GeoDataFrame(
        geometry=[union_stations.buffer(dist_etendue)],
        crs="EPSG:2154",
    )

    # 3. Distance minimale de chaque centroïde d'IRIS aux stations
    centroids = zone_2154.geometry.centroid
    zone_2154["dist_station_m"] = centroids.apply(
        lambda c: stations_2154.distance(c).min()
    )

    # 4. Classification des zones pour l'analyse économétrique
    zone_2154["groupe"] = zone_2154["dist_station_m"].apply(
        lambda d: (
            f"Zone Traitée (< {dist_traitee}m)"
            if d <= dist_traitee
            else f"Zone Contrôle (≥ {dist_traitee}m)"
        )
    )

    # 5. Retour au CRS d'origine pour le tracé
    return (
        zone_2154.to_crs(crs_orig),
        gdf_buf_traite.to_crs(crs_orig),
        gdf_buf_etendu.to_crs(crs_orig),
    )


# ==============================================================================
# 4. Données immobilières (DVF)
# ==============================================================================


def charger_dvf(code_insee="93070", dossier=None, debut=2018, fin=2026):
    """Charge les lignes DVF annuelles et les filtre sur la commune demandée.

    Les colonnes de coordonnées et de code communal sont harmonisées ; l'année
    et le nom du fichier source sont ajoutés. Les filtres de nettoyage et
    l'agrégation par mutation sont appliqués séparément par ``nettoyer_dvf``.
    """
    dossier = Path(dossier) if dossier is not None else DOSSIER_PROJET
    dfs = []
    code_insee = str(code_insee).strip().removesuffix(".0").zfill(5)

    for annee in range(int(debut), int(fin)):
        parquets = sorted(dossier.glob(f"**/*{annee}*.parquet"))
        csv_commune = sorted(dossier.glob(f"**/{code_insee}.{annee}.csv"))
        autres_csv = sorted(dossier.glob(f"**/*{annee}*.csv"))

        source = (
            parquets[0]
            if parquets
            else (
                csv_commune[0]
                if csv_commune
                else (autres_csv[0] if autres_csv else None)
            )
        )
        if source is None:
            print(f" {annee} : aucun fichier trouvé")
            continue

        df_temp = (
            pd.read_parquet(source)
            if source.suffix.lower() == ".parquet"
            else pd.read_csv(source, low_memory=False)
        )

        noms_normalises = {
            str(col).lower().replace("_", "").replace(" ", ""): col
            for col in df_temp.columns
        }
        col_com = next(
            (
                c
                for c in ("code_commune", "depcom", "code_insee", "insee_com")
                if c in df_temp.columns
            ),
            None,
        )
        if col_com is None:
            raise ValueError(
                f"{source} : aucune colonne de code commune trouvée. "
                f"Colonnes disponibles : {df_temp.columns.tolist()}"
            )

        if col_com != "code_commune":
            df_temp = df_temp.rename(columns={col_com: "code_commune"})

        for cible, alias in (
            ("longitude", ("longitude", "lon")),
            ("latitude", ("latitude", "lat")),
        ):
            if cible not in df_temp.columns:
                col = next(
                    (noms_normalises[nom] for nom in alias if nom in noms_normalises),
                    None,
                )
                if col is not None:
                    df_temp = df_temp.rename(columns={col: cible})

        if "longitude" not in df_temp.columns or "latitude" not in df_temp.columns:
            raise ValueError(
                f"{source} : colonnes longitude/latitude absentes. "
                f"Colonnes disponibles : {df_temp.columns.tolist()}"
            )

        codes_commune = (
            df_temp["code_commune"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
        )
        df_temp = df_temp.loc[codes_commune == code_insee].copy()

        if not df_temp.empty:
            df_temp["annee"] = annee
            df_temp["source_fichier"] = Path(source).name
            dfs.append(df_temp)
            print(f"✔ {source} : {len(df_temp)} lignes pour {annee}")

    if not dfs:
        raise FileNotFoundError(
            f"Aucune ligne DVF exploitable trouvée pour la commune {code_insee}."
        )

    df_dvf = pd.concat(dfs, ignore_index=True)
    for col in ("code_commune", "code_postal"):
        if col in df_dvf.columns:
            df_dvf[col] = (
                df_dvf[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(5)
            )

    return df_dvf


def nettoyer_dvf(
    df_dvf,
    prix_m2_min=2000,
    prix_m2_max=15000,
    surface_min=10,
    valeur_min=10000,
):
    """Nettoie des lignes DVF et produit une ligne par mutation résidentielle.

    Les filtres de nature, type de local, surface, valeur et prix au m² sont
    appliqués ici, après le chargement et le filtrage géographique.

    Le prix au m² est un proxy : la valeur totale de la mutation est divisée
    par la surface totale des seules lignes Appartement/Maison. La valeur peut
    donc inclure une dépendance ou un autre bien vendu dans la même mutation.
    """
    if not isinstance(df_dvf, pd.DataFrame):
        raise TypeError("df_dvf doit être un DataFrame pandas.")
    df_dvf = df_dvf.copy()

    colonnes_obligatoires = {
        "id_mutation",
        "nature_mutation",
        "type_local",
        "valeur_fonciere",
        "surface_reelle_bati",
        "longitude",
        "latitude",
        "annee",
        "source_fichier",
    }
    colonnes_absentes = sorted(colonnes_obligatoires - set(df_dvf.columns))
    if colonnes_absentes:
        raise ValueError(
            "Colonnes DVF obligatoires absentes : "
            f"{colonnes_absentes}. Colonnes disponibles : "
            f"{df_dvf.columns.tolist()}"
        )

    for c in ("valeur_fonciere", "surface_reelle_bati", "longitude", "latitude"):
        df_dvf[c] = pd.to_numeric(df_dvf[c], errors="coerce")
    if "date_mutation" in df_dvf.columns:
        df_dvf["date_mutation"] = pd.to_datetime(
            df_dvf["date_mutation"], format="mixed", errors="coerce"
        )

    logements = _aggreger_dvf_par_mutation(
        df_dvf,
        prix_m2_min=prix_m2_min,
        prix_m2_max=prix_m2_max,
        surface_min=surface_min,
        valeur_min=valeur_min,
    )

    print(
        "Filtre DVF terminé : "
        f"{len(logements)} mutations retenues sur {len(df_dvf)} lignes chargées."
    )
    return logements


def _aggreger_dvf_par_mutation(
    df_dvf,
    prix_m2_min=2000,
    prix_m2_max=15000,
    surface_min=10,
    valeur_min=10000,
):
    """Construit une observation par mutation à partir de ses lignes DVF.

    La valeur foncière est conservée une fois ; la surface additionne seulement
    les lignes résidentielles. Les coordonnées sont celles du premier logement
    résidentiel géolocalisé dans l'ordre des lignes source. Les lignes
    strictement identiques sont retirées avant agrégation.
    """
    colonnes_obligatoires = {
        "id_mutation",
        "nature_mutation",
        "type_local",
        "valeur_fonciere",
        "surface_reelle_bati",
        "longitude",
        "latitude",
        "annee",
        "source_fichier",
    }
    colonnes_absentes = sorted(colonnes_obligatoires - set(df_dvf.columns))
    if colonnes_absentes:
        raise ValueError(
            "Colonnes nécessaires à l'agrégation par mutation absentes : "
            f"{colonnes_absentes}"
        )

    doublons_exacts = int(df_dvf.duplicated().sum())
    donnees = df_dvf.drop_duplicates().copy()
    donnees["id_mutation"] = donnees["id_mutation"].astype("string").str.strip()
    donnees = donnees.dropna(subset=["id_mutation"])
    donnees = donnees.loc[donnees["id_mutation"].ne("")]
    donnees["nature_mutation"] = donnees["nature_mutation"].astype("string").str.strip()
    donnees["type_local"] = donnees["type_local"].astype("string").str.strip()

    colonnes_numeriques = (
        "valeur_fonciere",
        "surface_reelle_bati",
        "longitude",
        "latitude",
        "annee",
    )
    for col in colonnes_numeriques:
        donnees[col] = pd.to_numeric(donnees[col], errors="coerce")

    mutations = []
    valeurs_contradictoires = 0
    natures_contradictoires = 0
    surfaces_incompletes = 0
    sans_coordonnees = 0
    annees_contradictoires = 0
    dates_contradictoires = 0

    types_residentiels = {"Appartement", "Maison"}
    natures_vente = {"Vente", "Vente en l'état futur d'achèvement"}

    for id_mutation, lignes in donnees.groupby("id_mutation", sort=False):
        natures = lignes["nature_mutation"].dropna().unique()
        if len(natures) != 1:
            natures_contradictoires += 1
            continue
        nature = natures[0]
        if nature not in natures_vente:
            continue

        residentiel = lignes.loc[lignes["type_local"].isin(types_residentiels)].copy()
        if residentiel.empty:
            continue

        valeurs = lignes["valeur_fonciere"].dropna().unique()
        if len(valeurs) != 1:
            valeurs_contradictoires += 1
            continue
        valeur = float(valeurs[0])
        if valeur <= valeur_min:
            continue

        surfaces = residentiel["surface_reelle_bati"]
        if surfaces.isna().any() or surfaces.le(0).any():
            surfaces_incompletes += 1
            continue
        surface_residentielle = float(surfaces.sum())
        if surface_residentielle <= surface_min:
            continue

        types = sorted(residentiel["type_local"].dropna().unique().tolist())
        coordonnees_residentielles = residentiel.loc[
            residentiel["longitude"].between(-5.5, 8.5)
            & residentiel["latitude"].between(41.0, 51.5)
        ]
        if coordonnees_residentielles.empty:
            sans_coordonnees += 1
            continue

        annees = lignes["annee"].dropna().unique()
        if len(annees) != 1:
            annees_contradictoires += 1
            continue

        dates = (
            lignes["date_mutation"].dropna().unique()
            if "date_mutation" in lignes
            else []
        )
        if len(dates) > 1:
            dates_contradictoires += 1
            continue

        prix_m2 = valeur / surface_residentielle
        if not prix_m2_min <= prix_m2 <= prix_m2_max:
            continue

        representative = coordonnees_residentielles.iloc[0]
        ligne_sortie = {
            "id_mutation": id_mutation,
            "date_mutation": dates[0] if len(dates) else pd.NaT,
            "annee": int(annees[0]),
            "source_fichier": " | ".join(
                sorted(lignes["source_fichier"].dropna().astype(str).unique())
            ),
            "nature_mutation": nature,
            "type_local": types[0] if len(types) == 1 else "Mixte",
            "types_locaux_residentiels": " | ".join(types),
            "nb_lignes_residentielles": len(surfaces),
            "valeur_fonciere": valeur,
            "surface_reelle_bati": surface_residentielle,
            "prix_m2": prix_m2,
            "longitude": float(representative["longitude"]),
            "latitude": float(representative["latitude"]),
        }
        for col in ("code_commune", "code_postal", "nom_commune"):
            if col in lignes.columns:
                valeurs_colonne = lignes[col].dropna()
                ligne_sortie[col] = (
                    valeurs_colonne.iloc[0] if not valeurs_colonne.empty else pd.NA
                )
        mutations.append(ligne_sortie)

    colonnes_sortie = [
        "id_mutation",
        "date_mutation",
        "annee",
        "source_fichier",
        "nature_mutation",
        "type_local",
        "types_locaux_residentiels",
        "nb_lignes_residentielles",
        "valeur_fonciere",
        "surface_reelle_bati",
        "prix_m2",
        "longitude",
        "latitude",
    ]
    colonnes_sortie.extend(
        col
        for col in ("code_commune", "code_postal", "nom_commune")
        if col in donnees.columns
    )

    resultats = pd.DataFrame(mutations, columns=colonnes_sortie)
    if not resultats.empty and resultats["id_mutation"].duplicated().any():
        raise RuntimeError("L'agrégation DVF a produit plusieurs lignes par mutation.")

    print(
        "Agrégation DVF : "
        f"{doublons_exacts} lignes strictement identiques supprimées ; "
        f"{valeurs_contradictoires} mutations exclues pour valeur foncière "
        "absente/contradictoire ; "
        f"{natures_contradictoires} pour nature absente/contradictoire ; "
        f"{surfaces_incompletes} pour surface résidentielle manquante/invalide ; "
        f"{sans_coordonnees} sans coordonnée résidentielle valide ; "
        f"{annees_contradictoires} avec années contradictoires ; "
        f"{dates_contradictoires} avec dates contradictoires."
    )
    return resultats


def creer_data_immo_parquet(df_logements, chemin_sortie=None):
    """Enregistre en Parquet un DataFrame DVF déjà nettoyé."""
    if not isinstance(df_logements, pd.DataFrame):
        raise TypeError("df_logements doit être un DataFrame pandas.")

    colonnes_requises = {"id_mutation", "source_fichier", "annee"}
    colonnes_absentes = colonnes_requises - set(df_logements.columns)
    if colonnes_absentes:
        raise ValueError(
            "Le DataFrame doit contenir les colonnes d'un résultat nettoyé : "
            f"{sorted(colonnes_absentes)}."
        )
    if df_logements["id_mutation"].duplicated().any():
        raise ValueError(
            "Le DataFrame contient plusieurs lignes par id_mutation ; "
            "nettoyer_dvf doit être appelé avant l'enregistrement."
        )

    df_logements = df_logements.copy()
    chemin_sortie = (
        Path(chemin_sortie)
        if chemin_sortie is not None
        else DOSSIER_PROJET / "data" / "processed" / "data_immo.parquet"
    )
    if chemin_sortie.name in set(df_logements["source_fichier"].unique()):
        raise ValueError(
            "Le fichier de sortie ne peut pas porter le nom d'un fichier source."
        )

    for col in df_logements.select_dtypes(include=["object", "string"]).columns:
        df_logements[col] = (
            df_logements[col].astype("string").str.replace(r"\.0$", "", regex=True)
        )

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    df_logements.to_parquet(chemin_sortie, index=False)
    annees = sorted(df_logements["annee"].unique().tolist())
    print(
        f"Parquet créé : {chemin_sortie} "
        f"({len(df_logements)} lignes, années {annees})."
    )
    return chemin_sortie


def joindre_ventes_aux_iris(df_logements, zone_etude):
    """Joint chaque vente à un IRIS avec une clé textuelle commune."""
    colonnes_absentes = {"longitude", "latitude", "prix_m2"} - set(
        df_logements.columns
    )
    if colonnes_absentes:
        raise ValueError(
            f"Colonnes DVF absentes pour la jointure : {sorted(colonnes_absentes)}"
        )

    ventes = (
        df_logements.dropna(subset=["longitude", "latitude"])
        .drop(columns=[COL_IRIS, "index_right"], errors="ignore")
        .copy()
    )
    gdf_ventes = gpd.GeoDataFrame(
        ventes,
        geometry=gpd.points_from_xy(ventes["longitude"], ventes["latitude"]),
        crs="EPSG:4326",
    )
    if gdf_ventes.crs != zone_etude.crs:
        gdf_ventes = gdf_ventes.to_crs(zone_etude.crs)

    zone = zone_etude.copy()
    zone[COL_IRIS] = zone[COL_IRIS].astype(str)
    ventes_jointes = gpd.sjoin(
        gdf_ventes,
        zone[[COL_IRIS, "geometry"]],
        how="left",
        predicate="within",
    )
    return ventes_jointes.drop(columns=["index_right"], errors="ignore")


def agreger_dvf_par_iris(df_logements, zone_etude):
    """Agrège le prix médian au m² par IRIS après jointure spatiale."""
    ventes_avec_iris = joindre_ventes_aux_iris(df_logements, zone_etude)
    ventes_avec_iris = ventes_avec_iris[ventes_avec_iris[COL_IRIS].notna()].copy()

    dvf_iris = (
        ventes_avec_iris.groupby(COL_IRIS)["prix_m2"]
        .agg(prix_m2_median="median", nb_ventes="count")
        .reset_index()
    )
    dvf_iris[COL_IRIS] = dvf_iris[COL_IRIS].astype(str)

    zone_etude_copy = zone_etude.copy()
    zone_etude_copy[COL_IRIS] = zone_etude_copy[COL_IRIS].astype(str)
    return zone_etude_copy.merge(dvf_iris, on=COL_IRIS, how="left")


# ==============================================================================
# 5. Données socio-économiques (Filosofi, CSP, Ménages)
# ==============================================================================


def _trouver_excel_annee(annee: int, nom_prefere: str) -> pd.DataFrame:
    """Charge le fichier Excel Filosofi d'une année donnée."""
    chemin = DOSSIER_PROJET / "data" / "raw" / nom_prefere
    if not chemin.is_file():
        chemins_trouves = list(DOSSIER_PROJET.glob(f"**/{nom_prefere}"))
        if chemins_trouves:
            chemin = chemins_trouves[0]
        else:
            raise FileNotFoundError(
                f"Fichier Filosofi {annee} introuvable : {nom_prefere}"
            )
    return pd.read_excel(chemin, skiprows=5)


def _trouver_colonne_niveau_vie(df: pd.DataFrame, col_iris: str) -> str:
    """Identifie la colonne du niveau de vie médian dans un DataFrame Filosofi."""
    for col in ("DISP_MED18", "DISP_MED21", "DISP_MED"):
        if col in df.columns:
            return col

    for col in df.columns:
        if col != col_iris and any(mot in str(col).upper() for mot in ("MED", "DISP")):
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            if not serie.empty and 10000 <= serie.mean() <= 40000:
                return col

    for col in df.columns:
        if col != col_iris:
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            if not serie.empty and 15000 <= serie.mean() <= 35000:
                return col

    raise ValueError(
        "Impossible d'identifier la colonne de niveau de vie médian. "
        f"Colonnes : {list(df.columns)}"
    )


def _preparer_filosofi(df: pd.DataFrame, col_niv: str) -> pd.DataFrame:
    """Normalise l'identifiant IRIS sur 9 caractères et convertit la variable en numérique."""
    if "IRIS" not in df.columns:
        raise KeyError("La colonne 'IRIS' est absente du DataFrame.")

    out = df[["IRIS", col_niv]].rename(columns={"IRIS": COL_IRIS}).copy()
    out[COL_IRIS] = (
        out[COL_IRIS].astype(str).str.split(".").str[0].str.strip().str.zfill(9)
    )
    out[col_niv] = pd.to_numeric(out[col_niv], errors="coerce")
    return out


def charger_et_preparer_filosofi(
    annee: int,
    nom_fichier: str,
    zone_etude: gpd.GeoDataFrame,
):
    """Charge, prépare et fusionne les données Filosofi d'une année avec la zone d'étude.

    Returns
    -------
    tuple
        (GeoDataFrame fusionné, nom de la colonne de niveau de vie)
    """
    df_raw = _trouver_excel_annee(annee, nom_fichier)
    col_niv = _trouver_colonne_niveau_vie(df_raw, "IRIS")
    df_prep = _preparer_filosofi(df_raw, col_niv)

    zone_copie = zone_etude.copy()
    zone_copie[COL_IRIS] = zone_copie[COL_IRIS].astype(str).str.strip().str.zfill(9)

    df_fusion = zone_copie.merge(df_prep, on=COL_IRIS, how="inner")
    return df_fusion, col_niv


def preparer_csp(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et normalise les codes IRIS d'un tableau INSEE CSP."""
    col_iris_src = next(
        (c for c in df.columns if "IRIS" in str(c).upper() or "CODE" in str(c).upper()),
        df.columns[0],
    )
    out = df.copy()
    out[COL_IRIS] = (
        out[col_iris_src].astype(str).str.split(".").str[0].str.strip().str.zfill(9)
    )
    cols_num = {
        c: pd.to_numeric(out[c], errors="coerce")
        for c in out.columns
        if c != COL_IRIS
    }
    return pd.concat([out[[COL_IRIS]], pd.DataFrame(cols_num, index=out.index)], axis=1)


def calcul_indicateurs(df: pd.DataFrame, label_annee: str = "") -> pd.DataFrame:
    """Calcule la part des cadres et l'indice de mixité sociale (Shannon) depuis les CSP."""
    col_cadres = next(
        (
            c
            for c in df.columns
            if any(k in str(c).upper() for k in ("C11_ACT1564_CS3", "CS3", "GSEC13_23", "GSEC13"))
            and "ACT1564" in str(c).upper()
        ),
        None,
    )

    col_total = next(
        (
            c
            for c in df.columns
            if str(c).upper().startswith("C")
            and "ACT1564" in str(c).upper()
            and not any(x in str(c).upper() for x in ("CS", "GSEC", "_H", "_F", "STAT"))
        ),
        None,
    )
    if col_total is None:
        col_total = next(
            (
                c
                for c in df.columns
                if "ACT1564" in str(c).upper()
                and not any(x in str(c).upper() for x in ("CS", "GSEC", "_H", "_F", "STAT"))
            ),
            None,
        )

    if col_cadres is None or col_total is None:
        raise ValueError(
            f"[{label_annee}] Colonnes non trouvées. Cadres={col_cadres}, Total={col_total}."
        )

    part_cadres = (df[col_cadres] / df[col_total].replace(0, np.nan)) * 100

    gsec_codes = (
        "GSEC11_21",
        "GSEC12_22",
        "GSEC13_23",
        "GSEC14_24",
        "GSEC15_25",
        "GSEC16_26",
    )
    cols_csp = [
        c
        for c in df.columns
        if any(g in str(c).upper() for g in gsec_codes) and "ACT1564" in str(c).upper()
    ]
    if not cols_csp:
        cs_codes = ("CS1", "CS2", "CS3", "CS4", "CS5", "CS6")
        cols_csp = [
            c
            for c in df.columns
            if any(str(c).upper().endswith(k) for k in cs_codes)
            and "ACT1564" in str(c).upper()
        ]

    if len(cols_csp) >= 2:
        somme_csp = df[cols_csp].sum(axis=1).replace(0, np.nan)
        shannon = pd.Series(0.0, index=df.index)
        for c in cols_csp:
            p_i = (df[c] / somme_csp).clip(lower=0)
            masque = p_i > 0
            shannon.loc[masque] -= p_i.loc[masque] * np.log(p_i.loc[masque])
        mixite = shannon
    else:
        mixite = pd.Series(np.nan, index=df.index)

    base = df.loc[:, ~df.columns.duplicated()].drop(
        columns=["part_cadres", "mixité_sociale"], errors="ignore"
    )
    return pd.concat(
        [
            base,
            pd.DataFrame(
                {"part_cadres": part_cadres, "mixité_sociale": mixite},
                index=df.index,
            ),
        ],
        axis=1,
    )


def calcul_part_menages_cibles(df: pd.DataFrame, prefixe_annee: str) -> pd.DataFrame:
    """Part des ménages d'une personne et des couples sans enfant (en %), par IRIS."""
    col_men = f"{prefixe_annee}_MEN"
    cols = [col_men, f"{prefixe_annee}_MENPSEUL", f"{prefixe_annee}_MENCOUPSENF"]
    absentes = [c for c in cols if c not in df.columns]
    if absentes:
        raise KeyError(f"Colonnes introuvables : {absentes}")

    nb_cible = df[cols[1]].fillna(0) + df[cols[2]].fillna(0)
    part = nb_cible / df[col_men].replace(0, np.nan) * 100
    return pd.DataFrame(
        {COL_IRIS: df[COL_IRIS], "nb_cible": nb_cible, "part_cible_pct": part}
    )


def calcul_part_familles_cibles(df: pd.DataFrame, prefixe_annee: str) -> pd.DataFrame:
    """Part des familles monoparentales et des familles nombreuses (3+ enfants), en %."""
    col_men = f"{prefixe_annee}_MEN"
    col_mono = f"{prefixe_annee}_MENFAMMONO"
    absentes = [c for c in (col_men, col_mono) if c not in df.columns]
    if absentes:
        raise KeyError(f"Colonnes introuvables : {absentes}")

    col_fam3p = next(
        (
            c
            for c in df.columns
            if prefixe_annee in c
            and any(k in c.upper() for k in ("FAM3P", "FAM4P", "MENFAM3P"))
        ),
        None,
    )
    nb_cible = df[col_mono].fillna(0)
    if col_fam3p:
        nb_cible = nb_cible + df[col_fam3p].fillna(0)

    part = nb_cible / df[col_men].replace(0, np.nan) * 100
    return pd.DataFrame(
        {COL_IRIS: df[COL_IRIS], "nb_cible": nb_cible, "part_cible_pct": part}
    )


def calcul_taille_menages(df: pd.DataFrame, prefixe_annee: str) -> pd.DataFrame:
    """Nombre de ménages et nombre moyen de personnes par ménage, par IRIS."""
    col_men, col_pmen = f"{prefixe_annee}_MEN", f"{prefixe_annee}_PMEN"
    absentes = [c for c in (col_men, col_pmen) if c not in df.columns]
    if absentes:
        raise KeyError(f"Colonnes introuvables : {absentes}")

    return pd.DataFrame(
        {
            COL_IRIS: df[COL_IRIS],
            "nb_men": df[col_men],
            "taille_moy": df[col_pmen] / df[col_men].replace(0, np.nan),
        }
    )


def calculer_evolution(
    zone_etude,
    ind_debut,
    ind_fin,
    annee_debut,
    annee_fin,
    col_nb="nb_cible",
    col_part="part_cible_pct",
):
    """Joint deux indicateurs annuels aux IRIS et calcule l'évolution en points."""
    out = zone_etude[[COL_IRIS, "geometry"]].copy()
    out[COL_IRIS] = out[COL_IRIS].astype(str).str.strip().str.zfill(9)

    for ind, annee in ((ind_debut, annee_debut), (ind_fin, annee_fin)):
        out = out.merge(
            ind[[COL_IRIS, col_nb, col_part]].rename(
                columns={col_nb: f"nb_{annee}", col_part: f"part_{annee}"}
            ),
            on=COL_IRIS,
            how="left",
        )

    out["evol_pts"] = out[f"part_{annee_fin}"] - out[f"part_{annee_debut}"]
    return out


# ==============================================================================
# 6. Construction de logements (Sitadel)
# ==============================================================================


def _recuperer_ressource_departementale() -> dict:
    """Récupère les métadonnées de la ressource CSV départementale (data.gouv.fr)."""
    resp = requests.get(DATAGOUV_API_URL, timeout=30)
    resp.raise_for_status()
    ressource = next(
        (
            r
            for r in resp.json().get("resources", [])
            if "départementales" in (r.get("title") or "").lower()
            and (r.get("format") or "").lower() == "csv"
        ),
        None,
    )
    if not ressource:
        raise ValueError("Ressource CSV départementale introuvable sur data.gouv.fr.")
    return ressource


def _charger_via_api_tabulaire(resource_id, departement, annees):
    """Récupère les lignes via l'API tabulaire ; None en cas d'erreur."""
    annees_set = set(annees)
    dept_clean = departement.strip().lstrip("0") or "0"

    for dept_val in dict.fromkeys([departement.strip(), dept_clean]):
        rows, page, ok = [], 1, True
        while True:
            r = requests.get(
                f"{TABULAR_API_BASE}/{resource_id}/data/",
                params={
                    "page": page,
                    "page_size": 50,
                    "DEPARTEMENT_CODE__exact": dept_val,
                },
                timeout=30,
            )
            if not r.ok:
                ok = False
                break
            payload = r.json()
            batch = payload.get("data", [])
            if not batch:
                break
            for row in batch:
                annee = pd.to_numeric(row.get("ANNEE"), errors="coerce")
                if pd.notna(annee) and int(annee) in annees_set:
                    rows.append(row)
            if not payload.get("links", {}).get("next"):
                break
            page += 1
        if ok and rows:
            return pd.DataFrame(rows)
    return None


def charger_sitadel_api(departement=DEPARTEMENT, annees=ANNEES) -> pd.DataFrame:
    """Logements commencés par année (lignes) et type de logement (colonnes)."""
    ressource = _recuperer_ressource_departementale()
    df = _charger_via_api_tabulaire(ressource["id"], departement, annees)
    if df is None or df.empty:
        df = pd.read_csv(ressource["url"], sep=None, engine="python")

    df["ANNEE"] = pd.to_numeric(df["ANNEE"], errors="coerce")
    df["LOG_COM"] = pd.to_numeric(df["LOG_COM"], errors="coerce")
    df["DEPARTEMENT_CODE"] = df["DEPARTEMENT_CODE"].astype(str).str.strip().str.zfill(2)
    df = df[
        (df["DEPARTEMENT_CODE"] == departement.zfill(2))
        & df["ANNEE"].between(min(annees), max(annees))
    ]
    return df.pivot_table(
        index="ANNEE",
        columns="TYPE_LGT",
        values="LOG_COM",
        aggfunc="sum",
    )


# ==============================================================================
# 7. Visualisation
# ==============================================================================


def cadrer_carte(ax, gdf, marge=0.05):
    """Applique un cadrage avec marge autour d'une zone et masque les axes."""
    minx, miny, maxx, maxy = gdf.total_bounds
    mx = (maxx - minx) * marge
    my = (maxy - miny) * marge
    ax.set_xlim(minx - mx, maxx + mx)
    ax.set_ylim(miny - my, maxy + my)
    ax.set_axis_off()


def etiqueter_stations(ax, stations, fontsize=9.5):
    """Étiquette les stations (gère les variantes de nom de colonne)."""
    col_nom = next(
        (
            col
            for col in stations.columns
            if str(col).lower() in {"nom_zda", "nom_zda ", "nom", "libelle", "name"}
        ),
        None,
    )
    if col_nom is None:
        return

    for _, row in stations.iterrows():
        ax.text(
            row.geometry.x,
            row.geometry.y,
            row[col_nom],
            fontsize=fontsize,
            fontweight="bold",
            color="#111111",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )


def tracer_carte_evolution(
    evol,
    annee_debut,
    annee_fin,
    titre,
    unite_evol="points de %",
    col_evol="evol_pts",
):
    """Trace la carte d'évolution entre deux années (palette divergente centrée sur 0)."""
    vmax_evol = np.nanmax(np.abs(evol[col_evol]))
    vmax_evol = float(vmax_evol) if pd.notna(vmax_evol) and vmax_evol > 0 else 5.0

    fig, ax = plt.subplots(figsize=(9, 8))
    evol.plot(
        ax=ax,
        column=col_evol,
        cmap="RdBu_r",
        vmin=-vmax_evol,
        vmax=vmax_evol,
        edgecolor="black",
        linewidth=0.8,
        legend=True,
        missing_kwds={"color": "lightgrey", "edgecolor": "black", "hatch": "///"},
        legend_kwds={
            "label": f"Évolution ({unite_evol}, {annee_fin} - {annee_debut})",
            "orientation": "horizontal",
            "pad": 0.04,
        },
    )
    ax.set_title(
        f"Saint-Ouen-sur-Seine - Évolution ({annee_debut}-{annee_fin})\n{titre}",
        fontsize=12,
    )
    cadrer_carte(ax, evol)

    plt.tight_layout()
    plt.show()


def tracer_dynamique_construction(sitadel, annee_debut, annee_fin, label_dept):
    """Affiche les chiffres clés puis la courbe annuelle et l'indice base 100."""
    tous_log = sitadel["Tous Logements"]
    collectif = sitadel["Collectif et Residence"]
    indice = tous_log / tous_log.loc[annee_debut] * 100
    evol_pct = (tous_log.loc[annee_fin] / tous_log.loc[annee_debut] - 1) * 100

    print(
        f"Logements commencés ({label_dept}) - {annee_debut} : "
        f"{tous_log.loc[annee_debut]:,.0f}"
    )
    print(
        f"Logements commencés ({label_dept}) - {annee_fin} : "
        f"{tous_log.loc[annee_fin]:,.0f}"
    )
    print(f"Évolution {annee_debut} -> {annee_fin} : {evol_pct:+.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    axes[0].plot(
        tous_log.index,
        tous_log,
        marker="o",
        linewidth=2.2,
        color="#08519c",
        label="Tous logements commencés",
    )
    axes[0].plot(
        collectif.index,
        collectif,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="#3182bd",
        label="Collectif et résidence",
    )
    axes[0].set_title(
        f"Dynamique annuelle des mises en chantier ({annee_debut}-{annee_fin})\n{label_dept}",
        fontsize=12,
    )
    axes[0].set_ylabel("Nombre de logements commencés")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    couleurs = ["#d73027" if v >= 100 else "#4575b4" for v in indice]
    axes[1].bar(indice.index, indice, color=couleurs, edgecolor="black", alpha=0.85)
    axes[1].axhline(
        100,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Base 100 ({annee_debut})",
    )
    axes[1].set_title(
        f"Indice de dynamique de construction\nBase 100 en {annee_debut} (LOG_COM)",
        fontsize=12,
    )
    axes[1].set_ylabel(f"Indice ({annee_debut} = 100)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes:
        ax.set_xticks(list(tous_log.index))
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()