import os
import pandas as pd
import geopandas as gpd

def charger_carte_et_transports(code_insee="93070"):
    """Charge les IRIS, le tracé de la ligne 14 et les stations filtrées sur la commune."""
    # 1. Fond IRIS
    gdf_iris = gpd.read_file("IDF_iris.geojson")
    col_insee = 'insee_com' if 'insee_com' in gdf_iris.columns else 'depcom'
    zone_etude = gdf_iris[gdf_iris[col_insee].astype(str) == str(code_insee)].copy()

    # 2. Ligne 14
    ligne_14 = gpd.read_file("traces-des-lignes-de-transport-en-commun-idfm-2.geojson")
    if ligne_14.crs != zone_etude.crs:
        ligne_14 = ligne_14.to_crs(zone_etude.crs)

    # 3. Stations
    gdf_stations = gpd.read_file("emplacement-des-gares-idf.geojson")
    if gdf_stations.crs != zone_etude.crs:
        gdf_stations = gdf_stations.to_crs(zone_etude.crs)

    # Slicing spatial avec marge
    minx, miny, maxx, maxy = zone_etude.total_bounds
    marge_x = (maxx - minx) * 0.05
    marge_y = (maxy - miny) * 0.05
    stations_st_ouen = gdf_stations.cx[
        minx - marge_x : maxx + marge_x, 
        miny - marge_y : maxy + marge_y
    ].copy()

    return zone_etude, ligne_14, stations_st_ouen


def charger_et_nettoyer_dvf(code_insee="93070"):
    """Charge et harmonise l'ensemble des fichiers DVF (.txt et .csv)."""
    dfs = []
    mapping_txt = {
        "Date mutation": "date_mutation",
        "Nature mutation": "nature_mutation",
        "Valeur fonciere": "valeur_fonciere",
        "Type local": "type_local",
        "Surface reelle bati": "surface_reelle_bati",
        "Nombre pieces principales": "nombre_pieces_principales",
        "Code type local": "code_type_local",
    }

    # 1. Fichiers .txt (2017-2020)
    for annee in range(2017, 2021):
        fichier = f"valeursfoncieres-{annee}.txt"
        if os.path.exists(fichier):
            df_temp = pd.read_csv(fichier, sep="|", decimal=",", dtype={"Code departement": str, "Code commune": str}, low_memory=False)
            c_insee = df_temp["Code departement"].astype(str).str.zfill(2) + df_temp["Code commune"].astype(str).str.zfill(3)
            df_temp = df_temp[c_insee == str(code_insee)].copy()
            df_temp = df_temp.rename(columns=mapping_txt)
            df_temp["valeur_fonciere"] = pd.to_numeric(df_temp["valeur_fonciere"], errors="coerce")
            df_temp["surface_reelle_bati"] = pd.to_numeric(df_temp["surface_reelle_bati"], errors="coerce")
            dfs.append(df_temp)

    # 2. Fichiers .csv (2021-2025)
    for annee in range(2021, 2026):
        fichier = f"{code_insee}.{annee}.csv"
        if os.path.exists(fichier):
            df_temp = pd.read_csv(fichier, low_memory=False)
            for col in ["valeur_fonciere", "surface_reelle_bati"]:
                if col in df_temp.columns:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")
            dfs.append(df_temp)

    df_dvf = pd.concat(dfs, ignore_index=True)
    df_dvf["date_mutation"] = pd.to_datetime(df_dvf["date_mutation"], format="mixed", dayfirst=True)
    df_dvf["annee"] = df_dvf["date_mutation"].dt.year

    # Filtrage résidentiel & filtre anti-aberrants
    logements = df_dvf[
        (df_dvf["nature_mutation"].isin(["Vente", "Vente en l'état futur d'achèvement"]))
        & (df_dvf["type_local"].isin(["Appartement", "Maison"]))
        & (df_dvf["surface_reelle_bati"] > 10)
        & (df_dvf["valeur_fonciere"] > 10000)
    ].copy()

    logements["prix_m2"] = logements["valeur_fonciere"] / logements["surface_reelle_bati"]
    logements = logements[(logements["prix_m2"] >= 2000) & (logements["prix_m2"] <= 15000)]

    return logements


def agreger_dvf_par_iris(df_logements, zone_etude):
    """Effectue la jointure spatiale entre les ventes DVF et les IRIS."""
    gdf_ventes = gpd.GeoDataFrame(
        df_logements,
        geometry=gpd.points_from_xy(df_logements["longitude"], df_logements["latitude"]),
        crs="EPSG:4326",
    )

    if gdf_ventes.crs != zone_etude.crs:
        gdf_ventes = gdf_ventes.to_crs(zone_etude.crs)

    ventes_avec_iris = gpd.sjoin(gdf_ventes, zone_etude, how="inner", predicate="within")

    col_iris = "code_iris" if "code_iris" in zone_etude.columns else ("iris" if "iris" in zone_etude.columns else "CODE_IRIS")

    dvf_iris = (
        ventes_avec_iris.groupby(col_iris)["prix_m2"]
        .agg(prix_m2_median="median", nb_ventes="count")
        .reset_index()
    )

    zone_etude_copy = zone_etude.copy()
    zone_etude_copy[col_iris] = zone_etude_copy[col_iris].astype(str)
    dvf_iris[col_iris] = dvf_iris[col_iris].astype(str)

    return zone_etude_copy.merge(dvf_iris, on=col_iris, how="left")