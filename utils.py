# Imports et constantes

from pathlib import Path
import re
import unicodedata

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# Colonne IRIS utilisée dans les jointures et agrégations.
COL_IRIS = "code_iris"

DOSSIER_PROJET = Path(__file__).resolve().parent

# Périmètre IRIS

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
            f"Aucun IRIS trouvé pour la commune {code_insee} dans "
            "'insee_com'."
        )
    if zone_etude.crs is None:
        raise ValueError("Le fichier IRIS ne précise pas son système de coordonnées.")

    zone_etude[COL_IRIS] = zone_etude[COL_IRIS].astype("string")
    return zone_etude



# Ligne 14

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



# Proximité des gares

def calculer_zones_tampons(
    zone_etude, stations, dist_traitee=500, dist_etendue=800
):
    """Calcule la distance aux gares, les buffers et la classification
    Traité / Contrôle de chaque IRIS.

    Returns
    -------
    tuple : (IRIS avec 'dist_station_m' et 'groupe',
             GeoDataFrame buffer traité, GeoDataFrame buffer étendu)
    """
    crs_orig = zone_etude.crs

    # 1. Reprojection en Lambert-93 (EPSG:2154) pour des distances métriques
    zone_2154 = zone_etude.to_crs("EPSG:2154")
    stations_2154 = stations.to_crs("EPSG:2154")

    # 2. Géométries des buffers unifiés
    union_stations = unary_union(stations_2154.geometry)

    gdf_buf_traite = gpd.GeoDataFrame(
        geometry=[union_stations.buffer(dist_traitee)], crs="EPSG:2154"
    )
    gdf_buf_etendu = gpd.GeoDataFrame(
        geometry=[union_stations.buffer(dist_etendue)], crs="EPSG:2154"
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



# Données immobilières DVF

def charger_dvf(code_insee="93070", dossier=None):
    """Charge les lignes DVF annuelles et les filtre sur la commune demandée.

    Les colonnes de coordonnées et de code communal sont harmonisées; l'année
    et le nom du fichier source sont ajoutés. Les filtres de nettoyage et
    l'agrégation par mutation sont appliqués séparément par ``nettoyer_dvf``.
    """
    dossier = Path(dossier) if dossier is not None else DOSSIER_PROJET
    dfs = []
    code_insee = str(code_insee).strip().removesuffix(".0").zfill(5)

    for annee in range(2017, 2026):
        parquets = sorted(dossier.glob(f"**/*{annee}*.parquet"))
        csv_commune = sorted(dossier.glob(f"**/{code_insee}.{annee}.csv"))
        autres_csv = sorted(dossier.glob(f"**/*{annee}*.csv"))

        source = parquets[0] if parquets else (
            csv_commune[0] if csv_commune else (
                autres_csv[0] if autres_csv else None
            )
        )
        if source is None:
            print(f"⚠️ {annee} : aucun fichier trouvé")
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
            (c for c in ("code_commune", "depcom", "code_insee", "insee_com")
             if c in df_temp.columns),
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

    La valeur foncière est conservée une fois; la surface additionne seulement
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
        col for col in ("code_commune", "code_postal", "nom_commune")
        if col in donnees.columns
    )
    resultats = pd.DataFrame(mutations, columns=colonnes_sortie)
    if not resultats.empty and resultats["id_mutation"].duplicated().any():
        raise RuntimeError("L'agrégation DVF a produit plusieurs lignes par mutation.")

    print(
        "Agrégation DVF : "
        f"{doublons_exacts} lignes strictement identiques supprimées; "
        f"{valeurs_contradictoires} mutations exclues pour valeur foncière "
        "absente/contradictoire; "
        f"{natures_contradictoires} pour nature absente/contradictoire; "
        f"{surfaces_incompletes} pour surface résidentielle manquante/invalide; "
        f"{sans_coordonnees} sans coordonnée résidentielle valide; "
        f"{annees_contradictoires} avec années contradictoires; "
        f"{dates_contradictoires} avec dates contradictoires."
    )
    return resultats

def creer_data_immo_parquet(
    df_logements,
    chemin_sortie=None,
):
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
            "Le DataFrame contient plusieurs lignes par id_mutation; "
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
            df_logements[col]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
        )

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    df_logements.to_parquet(chemin_sortie, index=False)
    annees = sorted(df_logements["annee"].unique().tolist())
    print(
        f"Parquet créé : {chemin_sortie} "
        f"({len(df_logements)} lignes, années {annees})."
    )
    return chemin_sortie



# Jointure spatiale ventes–IRIS

def joindre_ventes_aux_iris(df_logements, zone_etude):
    """Joint chaque vente à un IRIS avec une clé textuelle commune."""
    colonnes_absentes = {"longitude", "latitude", "prix_m2"} - set(
        df_logements.columns
    )
    if colonnes_absentes:
        raise ValueError(
            "Colonnes DVF absentes pour la jointure : "
            f"{sorted(colonnes_absentes)}"
        )

    ventes = df_logements.dropna(subset=["longitude", "latitude"]).copy()
    gdf_ventes = gpd.GeoDataFrame(
        ventes,
        geometry=gpd.points_from_xy(
            ventes["longitude"],
            ventes["latitude"],
        ),
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
    ventes_jointes = ventes_jointes.drop(columns=["index_right"], errors="ignore")
    return ventes_jointes

def agreger_dvf_par_iris(df_logements, zone_etude):
    """Agrège le prix médian au m² par IRIS après jointure spatiale."""
    ventes_avec_iris = joindre_ventes_aux_iris(df_logements, zone_etude)
    ventes_avec_iris = ventes_avec_iris[
        ventes_avec_iris[COL_IRIS].notna()
    ].copy()

    dvf_iris = (
        ventes_avec_iris.groupby(COL_IRIS)["prix_m2"]
        .agg(prix_m2_median="median", nb_ventes="count")
        .reset_index()
    )
    dvf_iris[COL_IRIS] = dvf_iris[COL_IRIS].astype(str)

    zone_etude_copy = zone_etude.copy()
    zone_etude_copy[COL_IRIS] = zone_etude_copy[COL_IRIS].astype(str)
    return zone_etude_copy.merge(dvf_iris, on=COL_IRIS, how="left")



# Visualisation

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
            col for col in stations.columns
            if str(col).lower() in {
                "nom_zda",
                "nom_zda ",
                "nom",
                "libelle",
                "name",
            }
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
