import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# Colonne IRIS : on normalise son nom dès le chargement pour que
# tout le reste du code (et le notebook) utilise "code_iris".
COL_IRIS = "code_iris"

CANDIDATS_COL_IRIS = ("code_iris", "iris", "CODE_IRIS", "DCOMIRIS")


def _trouver_colonne_iris(gdf):
    """Renvoie le nom réel de la colonne IRIS dans un GeoDataFrame."""
    for col in CANDIDATS_COL_IRIS:
        if col in gdf.columns:
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


def charger_carte_et_transports(code_insee="93070"):
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
    gdf_iris = gpd.read_file("IDF_iris.geojson")
    col_insee = "insee_com" if "insee_com" in gdf_iris.columns else "depcom"
    zone_etude = gdf_iris[
        gdf_iris[col_insee].astype(str) == str(code_insee)
    ].copy()
    zone_etude = _normaliser_colonne_iris(zone_etude)

    # 2. Ligne 14
    ligne_14 = gpd.read_file(
        "traces-des-lignes-de-transport-en-commun-idfm-2.geojson"
    )
    if ligne_14.crs != zone_etude.crs:
        ligne_14 = ligne_14.to_crs(zone_etude.crs)

    # 3. Stations
    gdf_stations = gpd.read_file("emplacement-des-gares-idf.geojson")
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
):
    """Charge et harmonise les fichiers DVF (.txt 2017-2020 et .csv 2021-2025)
    puis filtre les ventes résidentielles non aberrantes.

    Parameters
    ----------
    code_insee : str
        Code INSEE de la commune.
    prix_m2_min, prix_m2_max : float
        Bornes de plausibilité du prix au m² (€/m²).
    surface_min : float
        Surface minimale du bâti (m²).
    valeur_min : float
        Valeur foncière minimale (€).

    Returns
    -------
    DataFrame : ventes de logements avec une colonne 'prix_m2' et 'annee'.
    """
    mapping_txt = {
        "Date mutation": "date_mutation",
        "Nature mutation": "nature_mutation",
        "Valeur fonciere": "valeur_fonciere",
        "Type local": "type_local",
        "Surface reelle bati": "surface_reelle_bati",
        "Nombre pieces principales": "nombre_pieces_principales",
        "Code type local": "code_type_local",
    }
    mutations_gardees = ["Vente", "Vente en l'état futur d'achèvement"]
    types_locaux = ["Appartement", "Maison"]

    dfs = []

    # 1. Fichiers .txt (2017-2020)
    for annee in range(2017, 2021):
        fichier = f"valeursfoncieres-{annee}.txt"
        if not os.path.exists(fichier):
            continue
        df_temp = pd.read_csv(
            fichier,
            sep="|",
            decimal=",",
            dtype={"Code departement": str, "Code commune": str},
            low_memory=False,
        )
        c_insee = (
            df_temp["Code departement"].astype(str).str.zfill(2)
            + df_temp["Code commune"].astype(str).str.zfill(3)
        )
        df_temp = df_temp[c_insee == str(code_insee)].copy()
        df_temp = df_temp.rename(columns=mapping_txt)
        dfs.append(df_temp)

    # 2. Fichiers .csv (2021-2025)
    for annee in range(2021, 2026):
        fichier = f"{code_insee}.{annee}.csv"
        if os.path.exists(fichier):
            dfs.append(pd.read_csv(fichier, low_memory=False))

    if not dfs:
        raise FileNotFoundError("Aucun fichier DVF trouvé pour ce code INSEE.")

    df_dvf = pd.concat(dfs, ignore_index=True)

    # Harmonisation des types
    df_dvf["valeur_fonciere"] = pd.to_numeric(
        df_dvf["valeur_fonciere"], errors="coerce"
    )
    df_dvf["surface_reelle_bati"] = pd.to_numeric(
        df_dvf["surface_reelle_bati"], errors="coerce"
    )
    df_dvf["date_mutation"] = pd.to_datetime(
        df_dvf["date_mutation"], format="mixed", dayfirst=True, errors="coerce"
    )
    df_dvf["annee"] = df_dvf["date_mutation"].dt.year

    # Filtrage résidentiel & anti-aberrants (seuils explicitement paramétrés,
    # à justifier dans le notebook)
    logements = df_dvf[
        df_dvf["nature_mutation"].isin(mutations_gardees)
        & df_dvf["type_local"].isin(types_locaux)
        & (df_dvf["surface_reelle_bati"] > surface_min)
        & (df_dvf["valeur_fonciere"] > valeur_min)
    ].copy()

    logements["prix_m2"] = (
        logements["valeur_fonciere"] / logements["surface_reelle_bati"]
    )
    logements = logements[
        logements["prix_m2"].between(prix_m2_min, prix_m2_max)
    ].copy()

    return logements

def agreger_dvf_par_iris(df_logements, zone_etude):
  """Agrège le prix médian DVF par IRIS.

  Priorise le code IRIS s'il est déjà présent dans DVF, sinon réalise une
  jointure spatiale.
  """
  zone_etude_copy = zone_etude.copy()
  col_iris = "code_iris"

  # 1. Vérifier si une colonne IRIS existe déjà dans df_logements
  cols_possibles = [
      "code_iris",
      "iris",
      "CODE_IRIS",
      "code_commune_iris",
      "code_insee_iris",
  ]
  col_dvf = next((c for c in cols_possibles if c in df_logements.columns), None)

  if col_dvf is not None:
    # Agrégation directe par code IRIS (rapide et sans bug de géométrie)
    df_temp = df_logements.copy()
    df_temp[col_iris] = df_temp[col_dvf].astype(str)
    dvf_iris = (
        df_temp.groupby(col_iris)["prix_m2"]
        .agg(prix_m2_median="median", nb_ventes="count")
        .reset_index()
    )
  else:
    # 2. Jointure spatiale uniquement si le code IRIS est absent
    df_clean = df_logements.dropna(subset=["longitude", "latitude"]).copy()
    gdf_ventes = gpd.GeoDataFrame(
        df_clean,
        geometry=gpd.points_from_xy(
            df_clean["longitude"], df_clean["latitude"]
        ),
        crs="EPSG:4326",
    )
    if gdf_ventes.crs != zone_etude_copy.crs:
      gdf_ventes = gdf_ventes.to_crs(zone_etude_copy.crs)

    ventes_avec_iris = gpd.sjoin(
        gdf_ventes, zone_etude_copy, how="inner", predicate="within"
    )
    dvf_iris = (
        ventes_avec_iris.groupby(col_iris)["prix_m2"]
        .agg(prix_m2_median="median", nb_ventes="count")
        .reset_index()
    )

  zone_etude_copy[col_iris] = zone_etude_copy[col_iris].astype(str)
  dvf_iris[col_iris] = dvf_iris[col_iris].astype(str)

  return zone_etude_copy.merge(dvf_iris, on=col_iris, how="left")


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
    col_nom = "nom_ZdA" if "nom_ZdA" in stations.columns else "nom_zda"
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