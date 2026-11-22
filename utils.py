from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# Colonne IRIS : on normalise son nom dès le chargement pour que
# tout le reste du code (et le notebook) utilise "code_iris".
COL_IRIS = "code_iris"

CANDIDATS_COL_IRIS = ("code_iris", "iris", "CODE_IRIS", "DCOMIRIS", "codeiris")
DOSSIER_PROJET = Path(__file__).resolve().parent


def _trouver_fichier(nom_fichier, dossier=None):
    """Recherche un fichier dans un dossier fourni, puis dans le dépôt."""
    candidats = []
    if dossier is not None:
        candidats.append(Path(dossier) / nom_fichier)
    candidats.extend([
        Path.cwd() / nom_fichier,
        DOSSIER_PROJET / nom_fichier,
        DOSSIER_PROJET / "data" / nom_fichier,
        DOSSIER_PROJET / "data" / "raw" / nom_fichier,
    ])

    for candidat in candidats:
        if candidat.is_file():
            return candidat

    # Le nom exact peut être dans un sous-dossier de données.
    recherche = DOSSIER_PROJET if dossier is None else Path(dossier)
    correspondances = sorted(recherche.glob(f"**/{nom_fichier}"))
    if correspondances:
        return correspondances[0]

    raise FileNotFoundError(
        f"Fichier introuvable : {nom_fichier}. "
        f"Répertoire recherché : {recherche}"
    )


def _trouver_colonne_iris(gdf):
    """Renvoie le nom réel de la colonne IRIS dans un GeoDataFrame."""
    for col in CANDIDATS_COL_IRIS:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if str(col).lower().replace("_", "").replace(" ", "") in {
            "codeiris",
            "iris",
            "dcomiris",
        }:
            return col
    raise ValueError(
        f"Aucune colonne IRIS trouvée. Colonnes disponibles : {list(gdf.columns)}"
    )


def _normaliser_colonne_iris(gdf):
    """Renomme la colonne IRIS (quelle que soit sa casse) en 'code_iris'."""
    col = _trouver_colonne_iris(gdf)
    if col != COL_IRIS:
        gdf = gdf.rename(columns={col: COL_IRIS})
    gdf[COL_IRIS] = gdf[COL_IRIS].astype(str)
    return gdf


def charger_carte_et_transports(code_insee="93070", dossier=None):
    """Charge les IRIS, le tracé de la ligne 14 et les stations filtrées sur la commune.

    Parameters
    ----------
    code_insee : str
        Code INSEE de la commune (défaut : Saint-Ouen, 93070).

    Returns
    -------
    tuple de GeoDataFrame : (IRIS de la commune, ligne 14, stations proches)
    """
    # 1. Fond IRIS
    gdf_iris = gpd.read_file(_trouver_fichier("IDF_iris.geojson", dossier))
    col_insee = next(
        (col for col in ("insee_com", "depcom", "code_commune", "insee")
         if col in gdf_iris.columns),
        None,
    )
    if col_insee is None:
        raise ValueError(
            "Code commune introuvable dans les IRIS. "
            f"Colonnes disponibles : {list(gdf_iris.columns)}"
        )
    zone_etude = gdf_iris[
        gdf_iris[col_insee].astype(str) == str(code_insee)
    ].copy()
    zone_etude = _normaliser_colonne_iris(zone_etude)

    # 2. Ligne 14
    ligne_14 = gpd.read_file(
        _trouver_fichier(
            "traces-des-lignes-de-transport-en-commun-idfm-2.geojson",
            dossier,
        )
    )
    if ligne_14.crs != zone_etude.crs:
        ligne_14 = ligne_14.to_crs(zone_etude.crs)

    # 3. Stations
    gdf_stations = gpd.read_file(
        _trouver_fichier("emplacement-des-gares-idf.geojson", dossier)
    )
    if gdf_stations.crs != zone_etude.crs:
        gdf_stations = gdf_stations.to_crs(zone_etude.crs)

    # Slicing spatial avec marge (stations dans et autour de la commune)
    minx, miny, maxx, maxy = zone_etude.total_bounds
    marge_x = (maxx - minx) * 0.05
    marge_y = (maxy - miny) * 0.05
    stations = gdf_stations.cx[
        minx - marge_x : maxx + marge_x, miny - marge_y : maxy + marge_y
    ].copy()

    return zone_etude, ligne_14, stations


def charger_et_nettoyer_dvf(
    code_insee="93070",
    prix_m2_min=2000,
    prix_m2_max=15000,
    surface_min=10,
    valeur_min=10000,
    dossier=None,
):
    """Charge et harmonise les fichiers DVF géolocalisés (parquet 2018-2020,
    CSV 2021-2025) puis filtre les ventes résidentielles non aberrantes.

    Returns
    -------
    DataFrame : ventes de logements avec 'prix_m2', 'annee', 'longitude', 'latitude'.
    """
    dossier = Path(dossier) if dossier is not None else DOSSIER_PROJET
    dfs = []
    sources_chargees = []
    code_insee = str(code_insee)

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

        df_temp = df_temp.rename(
            columns={
                "lon": "longitude",
                "lat": "latitude",
                "latitude": "latitude",
                "longitude": "longitude",
            }
        )
        noms_normalises = {
            str(col).lower().replace("_", "").replace(" ", ""): col
            for col in df_temp.columns
        }
        if "longitude" not in df_temp.columns:
            for cle in ("lon", "longitude"):
                if cle in noms_normalises:
                    df_temp = df_temp.rename(
                        columns={noms_normalises[cle]: "longitude"}
                    )
                    break
        if "latitude" not in df_temp.columns:
            for cle in ("lat", "latitude"):
                if cle in noms_normalises:
                    df_temp = df_temp.rename(
                        columns={noms_normalises[cle]: "latitude"}
                    )
                    break

        if "longitude" not in df_temp.columns or "latitude" not in df_temp.columns:
            print(
                f"⚠️ {source} : pas de longitude/latitude, ignoré. "
                f"Colonnes : {df_temp.columns.tolist()}"
            )
            continue

        # Filtre commune si le fichier contient une colonne commune
        col_com = next(
            (c for c in ("code_commune", "commune", "depcom",
                         "code_insee", "insee_com")
             if c in df_temp.columns),
            None,
        )
        if col_com is not None:
            df_temp = df_temp[
                df_temp[col_com].astype(str).str.zfill(5) == code_insee
            ].copy()

        if not df_temp.empty:
            df_temp["annee"] = annee
            dfs.append(df_temp)
            sources_chargees.append(str(source))
            print(f"✔ {source} : {len(df_temp)} lignes pour {annee}")

    if not dfs:
        raise FileNotFoundError("Aucun fichier DVF exploitable trouvé.")

    df_dvf = pd.concat(dfs, ignore_index=True)

    colonnes_obligatoires = {
        "nature_mutation",
        "type_local",
        "valeur_fonciere",
        "surface_reelle_bati",
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

    df_dvf["nature_mutation"] = df_dvf["nature_mutation"].astype(str).str.strip()
    df_dvf["type_local"] = df_dvf["type_local"].astype(str).str.strip()

    masque_base = (
        df_dvf["nature_mutation"].isin(
            ["Vente", "Vente en l'état futur d'achèvement"]
        )
        & df_dvf["type_local"].isin(["Appartement", "Maison"])
        & df_dvf["surface_reelle_bati"].gt(surface_min)
        & df_dvf["valeur_fonciere"].gt(valeur_min)
        & df_dvf["longitude"].between(-5.5, 8.5)
        & df_dvf["latitude"].between(41.0, 51.5)
    )
    logements = df_dvf.loc[masque_base].copy()
    logements["prix_m2"] = (
        logements["valeur_fonciere"] / logements["surface_reelle_bati"]
    )
    logements = logements[
        logements["prix_m2"].between(prix_m2_min, prix_m2_max)
    ].copy()

    print(
        "Filtre DVF terminé : "
        f"{len(logements)} ventes retenues sur {len(df_dvf)} lignes chargées."
    )
    return logements


def rattacher_ventes_aux_iris(df_logements, zone_etude):
    """Joint chaque vente à un IRIS avec une clé textuelle commune."""
    colonnes_absentes = {"longitude", "latitude", "prix_m2"} - set(
        df_logements.columns
    )
    if colonnes_absentes:
        raise ValueError(
            "Colonnes DVF absentes pour le rattachement : "
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
    rattachees = gpd.sjoin(
        gdf_ventes,
        zone[[COL_IRIS, "geometry"]],
        how="left",
        predicate="within",
    )
    rattachees = rattachees.drop(columns=["index_right"], errors="ignore")
    return rattachees


def agreger_dvf_par_iris(df_logements, zone_etude):
    """Agrège le prix médian au m² par IRIS après jointure spatiale."""
    ventes_avec_iris = rattacher_ventes_aux_iris(df_logements, zone_etude)
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


# ---------------------------------------------------------------------------
# Helpers de visualisation (réutilisés par plusieurs cartes du notebook)
# ---------------------------------------------------------------------------

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