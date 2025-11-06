import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import time

logger = logging.getLogger(__name__)


class DataFrameManager:
    """
    API evoluta:
    - load_data()
    - query_data(params) -> (df, xlabel, ylabel, meta)
    - query_data_for_map(params) -> (df_agg, xlabel, ylabel, meta)
    - dataset_meta() -> dict
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df: Optional[pd.DataFrame] = None

    # ---------- Public ----------

    def load_data(self):
        """
        Carica il dataset principale (preferendo il CSV) e misura il tempo di caricamento.
        """
        candidates = [
            os.path.join(self.data_dir, "df_ridotto_bot.csv"),    # ✅ preferisci CSV
            os.path.join(self.data_dir, "df_ridotto_bot.xlsx"),
            os.path.join("resources", "df_ridotto_bot.csv"),
            os.path.join("resources", "df_ridotto_bot.xlsx"),
        ]

        last_err = None
        for p in candidates:
            if os.path.exists(p):
                try:
                    start = time.time()
                    if p.lower().endswith(".xlsx"):
                        df = pd.read_excel(p, engine="openpyxl")
                    else:
                        df = pd.read_csv(p, low_memory=False)
                    elapsed = round(time.time() - start, 2)

                    self.df = self._normalize_df(df)
                    logger.info(
                        f"✅ Dati caricati da: {p} | righe={len(self.df)} | colonne={len(self.df.columns)} | tempo={elapsed}s"
                    )
                    print(f"📊 Dataset caricato da: {p}\n   → {len(self.df):,} righe, {len(self.df.columns)} colonne (in {elapsed}s)")
                    return
                except Exception as e:
                    last_err = e
                    logger.error(f"❌ Errore caricamento {p}: {e}")

        if last_err:
            raise last_err
        raise FileNotFoundError("❌ Nessun dataset trovato (né CSV né XLSX) nei percorsi attesi.")

    def dataset_meta(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        latest_year = None
        if "anno" in self.df.columns and self.df["anno"].notna().any():
            latest_year = int(pd.to_numeric(self.df["anno"], errors="coerce").dropna().max())
        sources = self._infer_sources(self.df.columns.tolist())
        coverage = f"{len(self.df):,}x{len(self.df.columns)}"
        return {
            "latest_year": latest_year,
            "sources": sources,
            "coverage_str": coverage
        }

    def query_data(self, params) -> Tuple[pd.DataFrame, str, str, Dict[str, Any]]:
        """
        Flusso:
          1) copia df
          2) alias & basi
          3) filtri (comuni, periodo con default)
          4) derived metrics (formula/per_capita/share_of + YoY/CAGR)
          5) casi speciali (distribution)
          6) verifica metriche richieste
          7) aggregazione + normalizzazione
          8) label assi + meta (fonti/anno/coverage)
        """
        if self.df is None:
            raise RuntimeError("Dati non caricati: chiama load_data() prima di query_data().")

        df = self.df.copy()

        # 1) alias minimi utili
        self._add_aliases(df)

        # 2) filtri standard
        df = self._filter_comuni(df, params)
        df = self._filter_period(df, params)

        # 3) derived
        df = self._add_derived_metrics(df, params)
        df = self._add_analytics(df, params)  # YoY, CAGR, rank, ecc.

        # 4) caso speciale: distribuzione per classi di reddito
        qtype_tmp = self._query_type_value(params)
        if qtype_tmp == "distribution":
            out, x, y = self._handle_distribution(df, params)
            meta = self._meta(df, params)
            return out, x, y, meta

        # 5) metriche richieste
        requested, available = self._requested_vs_available(df, params)
        if not available:
            logger.warning(f"Metriche richieste non trovate: {requested}")
            return pd.DataFrame(), "Comune o Anno", "", self._meta(df, params)

        # 6) aggregazione & normalizzazione
        df_out, group = self._aggregate(df, params, available)

        xlabel = group.capitalize() if group else "Comune o Anno"
        ylabel = self._friendly_ylabel(df_out, group)
        return df_out, xlabel, ylabel, self._meta(df, params)

    def query_data_for_map(self, params) -> Tuple[pd.DataFrame, str, str, Dict[str, Any]]:
        """Restituisce df aggregato a livello regionale (se possibile) o comunale per mappa."""
        if self.df is None:
            raise RuntimeError("Dati non caricati.")
        df = self.df.copy()
        self._add_aliases(df)
        df = self._filter_period(df, params)

        metric = (params.metrics or ["average_income"])[0]
        # Try region-level
        if "regione_norm" in df.columns:
            df_map = df.groupby("regione_norm", dropna=False)[[metric]].sum(numeric_only=True).reset_index()
            meta = self._meta(df, params)
            meta["map_level"] = "regione"
            return df_map, "Regione", metric, meta
        # Fallback comune
        if "comune" in df.columns:
            df_map = df.groupby("comune", dropna=False)[[metric]].sum(numeric_only=True).reset_index()
            meta = self._meta(df, params)
            meta["map_level"] = "comune"
            return df_map, "Comune", metric, meta
        return pd.DataFrame(), "", "", self._meta(df, params)

    # ---------- Helpers: Base ----------

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "anno" in df.columns:
            df["anno"] = pd.to_numeric(df["anno"], errors="coerce").astype("Int64")
        # Coerenza popolazione totale
        if "pop_totale" not in df.columns and "popolazione" in df.columns:
            df["pop_totale"] = pd.to_numeric(df["popolazione"], errors="coerce")
        return df

    def _add_aliases(self, df: pd.DataFrame) -> None:
        if "total_income" not in df.columns and "reddito_imponibile_ammontare_in_euro" in df.columns:
            df["total_income"] = pd.to_numeric(df["reddito_imponibile_ammontare_in_euro"], errors="coerce")
        if (
            "average_income" not in df.columns
            and "total_income" in df.columns
            and "numero_contribuenti" in df.columns
        ):
            df["average_income"] = df["total_income"] / pd.to_numeric(df["numero_contribuenti"], errors="coerce")

    def _filter_comuni(self, df: pd.DataFrame, params) -> pd.DataFrame:
        if getattr(params, "comuni", None) and "comune" in df.columns:
            comuni_req = [str(c).strip().lower() for c in params.comuni]
            before = len(df)
            out = df[df["comune"].str.lower().isin(comuni_req)]
            logger.info(f"Filtro comuni {params.comuni} -> {len(out)}/{before} righe")
            return out
        return df

    def _filter_period(self, df: pd.DataFrame, params) -> pd.DataFrame:
        """
        Gestisce il filtro per anno o intervallo.
        """
        if "anno" not in df.columns:
            return df

        if getattr(params, "anno", None) is not None:
            return df[df["anno"] == params.anno]

        if getattr(params, "start_year", None) is not None and getattr(params, "end_year", None) is not None:
            return df[(df["anno"] >= params.start_year) & (df["anno"] <= params.end_year)]

        qtype_tmp = self._query_type_value(params)
        if qtype_tmp == "time_series":
            return df

        latest = int(pd.to_numeric(df["anno"], errors="coerce").dropna().max())
        out = df[df["anno"] == latest]
        try:
            if getattr(params, "anno", None) is None:
                params.anno = latest
        except Exception:
            pass
        return out

    def _latest_year(self, df: pd.DataFrame) -> int:
        return int(pd.to_numeric(df["anno"], errors="coerce").dropna().max())

    def _query_type_value(self, params) -> str:
        return params.query_type.value if hasattr(params.query_type, "value") else str(params.query_type)

    # ---------- Helpers: Derived metrics ----------

    def _add_derived_metrics(self, df: pd.DataFrame, params) -> pd.DataFrame:
        derived = getattr(params, "derived_metrics", []) or []
        if not derived:
            return df

        handlers = {
            "per_capita": self._derive_per_capita,
            "share_of": self._derive_share_of,
        }

        for d in derived:
            name = d.get("name")
            if not name:
                continue

            formula = d.get("formula")
            if formula:
                try:
                    df[name] = df.eval(formula)
                    continue
                except Exception as e:
                    logger.warning(f"Errore nella formula '{name}': {formula} -> {e}")

            dtype = (d.get("type") or "").lower()
            try:
                handler = handlers.get(dtype)
                if handler:
                    df = handler(df, name, d)
            except Exception as e:
                logger.warning(f"Errore derived metric '{name}' di tipo '{dtype}': {e}")

        return df

    def _derive_per_capita(self, df: pd.DataFrame, name: str, spec: Dict[str, Any]) -> pd.DataFrame:
        base = spec.get("metric") or spec.get("base")
        if base and base in df.columns and "pop_totale" in df.columns:
            df[name] = pd.to_numeric(df[base], errors="coerce") / pd.to_numeric(df["pop_totale"], errors="coerce")
        return df

    def _derive_share_of(self, df: pd.DataFrame, name: str, spec: Dict[str, Any]) -> pd.DataFrame:
        num = spec.get("numerator") or spec.get("num")
        den = spec.get("denominator") or spec.get("den")
        if num in df.columns and den in df.columns:
            df[name] = pd.to_numeric(df[num], errors="coerce") / pd.to_numeric(df[den], errors="coerce")
        return df

    # ---------- Analytics (YoY, CAGR, rank) ----------

    def _add_analytics(self, df: pd.DataFrame, params) -> pd.DataFrame:
        """Regole semplici per generare analisi derivate quando richieste via metrics (yoy_*, cagr_*)."""
        metrics = list(params.metrics or [])
        if not metrics:
            return df

        def _safe_sort(d: pd.DataFrame) -> pd.DataFrame:
            if "anno" in d.columns:
                try:
                    return d.sort_values("anno")
                except Exception:
                    return d
            return d

        dfx = df.copy()
        if "anno" not in dfx.columns:
            return dfx

        for m in list(metrics):
            if m.startswith("yoy_"):
                base = m[4:]
                if base in dfx.columns:
                    dfx[m] = _safe_sort(dfx).groupby("comune", dropna=False)[base].pct_change() * 100.0
            if m.startswith("cagr_"):
                base = m[5:]
                if base in dfx.columns:
                    # CAGR calcolato per comune sull'intero intervallo disponibile (approx)
                    dfx = dfx.sort_values(["comune","anno"])
                    def _cagr(g):
                        n = g["anno"].dropna().nunique()
                        if n <= 1: return np.nan
                        v0 = pd.to_numeric(g[base], errors="coerce").iloc[0]
                        v1 = pd.to_numeric(g[base], errors="coerce").iloc[-1]
                        if v0 in (0, None, np.nan): return np.nan
                        return ( (v1 / v0) ** (1/(n-1)) - 1 ) * 100.0
                    dfx[m] = dfx.groupby("comune", dropna=False).apply(_cagr).reset_index(level=0, drop=True)
        return dfx

    # ---------- Helpers: Distribuzione redditi ----------

    def _handle_distribution(self, df: pd.DataFrame, params) -> Tuple[pd.DataFrame, str, str]:
        if not getattr(params, "comuni", None):
            logger.warning("Distribuzione: nessun comune specificato")
            return pd.DataFrame(), "Classe reddito", "Ammontare (euro)"

        cols = [
            "reddito_complessivo_da_0_a_10000_euro_ammontare_in_euro",
            "reddito_complessivo_da_10000_a_15000_euro_ammontare_in_euro",
            "reddito_complessivo_da_15000_a_26000_euro_ammontare_in_euro",
            "reddito_complessivo_da_26000_a_55000_euro_ammontare_in_euro",
            "reddito_complessivo_da_55000_a_75000_euro_ammontare_in_euro",
            "reddito_complessivo_da_75000_a_120000_euro_ammontare_in_euro",
            "reddito_complessivo_oltre_120000_euro_ammontare_in_euro",
        ]
        label_map = {
            cols[0]: "Da 0 a 10 mila",
            cols[1]: "Da 10 mila a 15 mila",
            cols[2]: "Da 15 mila a 26 mila",
            cols[3]: "Da 26 mila a 55 mila",
            cols[4]: "Da 55 mila a 75 mila",
            cols[5]: "Da 75 mila a 120 mila",
            cols[6]: "Oltre 120 mila",
        }

        comuni_list = list(params.comuni)

        if len(comuni_list) == 1:
            comune = str(comuni_list[0]).strip().lower()
            dfc = df[df["comune"].str.lower() == comune] if "comune" in df.columns else df.copy()
            available_cols = [c for c in cols if c in dfc.columns]
            if not available_cols:
                logger.warning("Distribuzione: colonne reddito per classi non trovate")
                return pd.DataFrame(), "Classe reddito", "Ammontare (euro)"
            values = [float(pd.to_numeric(dfc[c], errors="coerce").sum()) for c in available_cols]
            classes = [label_map.get(c, c) for c in available_cols]
            out = pd.DataFrame({"anno": dfc["anno"].max() if "anno" in dfc.columns else None,
                                "classe": classes, "valore": values})
            return out, "Classe reddito", "Ammontare (euro)"

        if "comune" not in df.columns:
            logger.warning("Distribuzione: colonna 'comune' mancante")
            return pd.DataFrame(), "Classe reddito", "Ammontare (euro)"

        dfd = df.copy()
        if "anno" in dfd.columns and dfd["anno"].notna().any():
            try:
                latest = int(pd.to_numeric(dfd["anno"], errors="coerce").dropna().max())
                dfd = dfd[dfd["anno"] == latest]
            except Exception:
                pass
        comuni_req = [str(c).strip().lower() for c in comuni_list]
        rows = []
        for col in cols:
            if col not in dfd.columns:
                continue
            row = {"classe": label_map.get(col, col)}
            for cm in comuni_req:
                val = float(pd.to_numeric(dfd[dfd["comune"].str.lower() == cm][col], errors="coerce").sum())
                row[cm] = val
            rows.append(row)
        if not rows:
            logger.warning("Distribuzione: nessuna classe disponibile per confronto")
            return pd.DataFrame(), "Classe reddito", "Ammontare (euro)"
        out_df = pd.DataFrame(rows)
        return out_df, "Classe reddito", "Ammontare (euro)"

    # ---------- Helpers: Aggregation ----------

    def _requested_vs_available(self, df: pd.DataFrame, params) -> tuple[List[str], List[str]]:
        derived = getattr(params, "derived_metrics", []) or []
        requested = list(
            set([str(m).strip().lower() for m in (params.metrics or [])] + [d["name"] for d in derived if "name" in d])
        )
        available = [m for m in requested if m in df.columns]
        return requested, available

    def _aggregate(self, df: pd.DataFrame, params, available: List[str]) -> tuple[pd.DataFrame, Optional[str]]:
        group = getattr(params, "groupby", None)
        qtype_tmp = self._query_type_value(params)
        has_year = "anno" in df.columns
        has_comune = "comune" in df.columns

        if not group:
            if qtype_tmp == "time_series" and has_year:
                group = "anno"
            elif qtype_tmp == "compare_comuni" and has_comune:
                if has_year and not getattr(params, "anno", None) and not getattr(params, "start_year", None):
                    latest = self._latest_year(df)
                    before = len(df)
                    df = df[df["anno"] == latest]
                    logger.info(f"Default anno (compare): {latest} -> {len(df)}/{before} righe")
                group = "comune"
            else:
                group = "anno" if has_year else ("comune" if has_comune else None)

        normalize = bool(getattr(params, "normalize_by_population", False))
        as_percent = bool(getattr(params, "normalize_as_percent", False))

        if not group:
            return df[available], None

        multiple_comuni = bool(getattr(params, "comuni", None)) and len(params.comuni) > 1
        agg_cols = list(set(available + (["pop_totale"] if normalize and "pop_totale" in df.columns else [])))

        if group == "anno" and multiple_comuni and "comune" in df.columns:
            keys = ["anno", "comune"]
            df_g = df.groupby(keys, dropna=False)[agg_cols].sum(numeric_only=True).reset_index()
            if normalize and "pop_totale" in df_g.columns:
                out_cols = []
                for m in available:
                    out_name = f"{m}_pct_pop" if as_percent else f"{m}_per_capita"
                    df_g[out_name] = (
                        pd.to_numeric(df_g[m], errors="coerce") / pd.to_numeric(df_g["pop_totale"], errors="coerce")
                    ) * (100.0 if as_percent else 1.0)
                    out_cols.append(out_name)
                if len(out_cols) == 1:
                    tmp = df_g[["anno", "comune", out_cols[0]]]
                    df_out = tmp.pivot(index="anno", columns="comune", values=out_cols[0]).reset_index().fillna(0)
                    return df_out, "anno"
                cols_keep = ["anno", "comune"] + out_cols
                return df_g[cols_keep], "anno"

            if len(available) == 1:
                metric = available[0]
                tmp = df_g[["anno", "comune", metric]]
                df_out = tmp.pivot(index="anno", columns="comune", values=metric).reset_index().fillna(0)
                return df_out, "anno"

            return df.groupby("anno", dropna=False)[available].sum(numeric_only=True).reset_index(), "anno"

        df_g = df.groupby(group, dropna=False)[agg_cols].sum(numeric_only=True).reset_index()
        if normalize and "pop_totale" in df_g.columns:
            out_cols = []
            for m in available:
                out_name = f"{m}_pct_pop" if as_percent else f"{m}_per_capita"
                df_g[out_name] = (
                    pd.to_numeric(df_g[m], errors="coerce") / pd.to_numeric(df_g["pop_totale"], errors="coerce")
                ) * (100.0 if as_percent else 1.0)
                out_cols.append(out_name)
            keep = [group] + out_cols
            return df_g[keep], group

        return df_g[[group] + available], group

    # ---------- Meta ----------
    def _meta(self, df: pd.DataFrame, params) -> Dict[str, Any]:
        sources = self._infer_sources(df.columns.tolist())
        latest_year = None
        if "anno" in df.columns and df["anno"].notna().any():
            latest_year = int(pd.to_numeric(df["anno"], errors="coerce").dropna().max())
        return {
            "sources": sources,
            "latest_year": latest_year,
            "coverage_str": f"{len(df):,}x{len(df.columns)}"
        }

    def _infer_sources(self, cols: List[str]) -> List[str]:
        out = set()
        for c in cols:
            cl = c.lower()
            if "reddito" in cl or "imposta" in cl or "irpef" in cl:
                out.add("MEF")
            if "pop" in cl or "laureati" in cl or "saldo_migratorio" in cl:
                out.add("ISTAT")
            if "impres" in cl or "brevetti" in cl:
                out.add("Infocamere/Brevetti")
        return sorted(out)

    # ---------- Extras ----------

    def sources_for_metrics(self, metric_list: List[str]) -> List[str]:
        return ["MEF" if "reddito" in m else "ISTAT" for m in metric_list]

    def list_comuni(self) -> List[str]:
        if self.df is None or "comune" not in self.df.columns:
            return []
        return sorted({str(c).strip() for c in self.df["comune"].dropna().unique()})

    # Friendly ylabel builder
    def _friendly_ylabel(self, df_out: pd.DataFrame, group: Optional[str]) -> str:
        # Heuristica minimale: percentuali e per-capita
        cols = [c for c in df_out.columns if c != (group or "")]
        label = " / ".join(cols)
        if any(str(c).endswith("_pct_pop") for c in cols):
            return "% popolazione"
        if any("income" in str(c) or "euro" in str(c) for c in cols):
            return "Valore (€)"
        return label
