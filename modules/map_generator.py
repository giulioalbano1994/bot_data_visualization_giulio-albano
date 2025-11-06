import io
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    gpd = None


class MapGenerator:
    """
    Genera coropleta regionale o comunale.
    Richiede uno shapefile/geojson configurato via ENV:
    - IT_REGIONI_GEOJSON (regioni)
    - IT_COMUNI_GEOJSON (comuni) [opzionale]
    Colonne attese: 'regione_norm' o 'comune' per join nominale.
    """

    def __init__(self):
        self.reg_path = os.getenv("IT_REGIONI_GEOJSON", "")
        self.com_path = os.getenv("IT_COMUNI_GEOJSON", "")

    def generate_choropleth(self, df: pd.DataFrame, metric_col: str, level: str = "regione",
                            title: str = "", subtitle: str = "") -> bytes:
        if gpd is None:
            # Fallback: heatmap fake (bar chart sorted) se geopandas mancante
            df_sorted = df.sort_values(metric_col, ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(df_sorted.iloc[:, 0], df_sorted[metric_col])
            ax.set_title(title)
            ax.set_xlabel(metric_col)
            ax.invert_yaxis()
            if subtitle:
                plt.suptitle(subtitle, y=0.98, fontsize=9, color="#444")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            plt.close()
            return buf.getvalue()

        # Geopandas path
        if level == "regione" and self.reg_path and os.path.exists(self.reg_path):
            gdf = gpd.read_file(self.reg_path)
            key_left = gdf.columns[0]  # we will try to guess
            # Try to normalize name column
            name_cols = [c for c in gdf.columns if "nome" in c.lower() or "reg" in c.lower()]
            geo_name = name_cols[0] if name_cols else gdf.columns[0]
            # Merge
            left = gdf.rename(columns={geo_name: "regione_norm"})
            right = df.rename(columns={df.columns[0]: "regione_norm"})
            mg = left.merge(right, on="regione_norm", how="left")
        elif level == "comune" and self.com_path and os.path.exists(self.com_path):
            gdf = gpd.read_file(self.com_path)
            name_cols = [c for c in gdf.columns if "comune" in c.lower() or "nome" in c.lower()]
            geo_name = name_cols[0] if name_cols else gdf.columns[0]
            left = gdf.rename(columns={geo_name: "comune"})
            right = df.rename(columns={df.columns[0]: "comune"})
            mg = left.merge(right, on="comune", how="left")
        else:
            # Fallback se file mancanti
            df_sorted = df.sort_values(metric_col, ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(df_sorted.iloc[:, 0], df_sorted[metric_col])
            ax.set_title(title)
            ax.set_xlabel(metric_col)
            ax.invert_yaxis()
            if subtitle:
                plt.suptitle(subtitle, y=0.98, fontsize=9, color="#444")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            plt.close()
            return buf.getvalue()

        # Plot
        fig, ax = plt.subplots(figsize=(8.8, 9.2))
        mg.plot(column=metric_col, ax=ax, legend=True, cmap="viridis", missing_kwds={"color": "lightgrey"})
        ax.set_axis_off()
        ax.set_title(title)
        if subtitle:
            plt.suptitle(subtitle, y=0.98, fontsize=9, color="#444")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close()
        return buf.getvalue()
