import os
import re
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


# -----------------------------
# Enums & Dataclasses (API STABILI)
# -----------------------------
class QueryType(Enum):
    SINGLE_COMUNE = "single_comune"
    COMPARE_COMUNI = "compare_comuni"
    TOP_COMUNI = "top_comuni"
    TIME_SERIES = "time_series"
    STATISTICS = "statistics"
    REDDITO_ANALYSIS = "reddito_analysis"
    PROVINCIA_DATA = "provincia_data"
    REGIONE_DATA = "regione_data"
    DISTRIBUTION = "distribution"


class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"


@dataclass
class QueryParameters:
    query_type: QueryType
    chart_type: ChartType
    comuni: Optional[List[str]] = None
    provincia: Optional[str] = None
    regione: Optional[str] = None
    metrics: Optional[List[str]] = None
    anno: Optional[int] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    n_results: Optional[int] = 10
    ascending: Optional[bool] = False
    groupby: Optional[str] = None
    derived_metrics: Optional[List[Dict[str, Any]]] = None
    normalize_by_population: Optional[bool] = None
    normalize_as_percent: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        if isinstance(self.query_type, QueryType):
            d["query_type"] = self.query_type.value
        if isinstance(self.chart_type, ChartType):
            d["chart_type"] = self.chart_type.value
        return d


# -----------------------------
# LLM Processor
# -----------------------------
class LLMProcessor:
    """
    Modulare: build_prompt â†’ call_llm â†’ parse_response â†’ normalize/fallback.
    Mantiene i metodi pubblici usati da main.py:
    - available_variables(limit)
    - process_request(text) -> QueryParameters
    - generate_commentary(df, params) -> str
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # Catalogo variabili (best-effort da piÃ¹ candidati)
        self.variable_catalog = self._load_variable_catalog()
        self.metric_mapping = self._build_metric_mapping(self.variable_catalog)
        self.catalog_text = self._catalog_preview(self.variable_catalog)
        self.examples_text = self._examples_text()

    # ---------- Public API ----------

    def available_variables(self, limit: int = 30) -> List[str]:
        seen, out = set(), []
        for _, r in self.variable_catalog.iterrows():
            lvl = str(r.get("livello", "")).lower()
            if any(x in lvl for x in ["identificativo", "tecnico", "da escludere"]):
                continue
            v = str(r.get("variabile", "")).strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
            if len(out) >= limit:
                break
        return out

    def process_request(self, user_request: str) -> QueryParameters:
        """
        Restituisce sempre un QueryParameters valido.
        """
        try:
            # Fallback locale se manca la chiave OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("No OpenAI key, using local fallback")
                return self._fallback_parameters(user_request)
            system, user = self._build_prompt(user_request)
            raw = self._call_llm(system, user)
            params = self._parse_llm_response(raw)
            # Normalizzazioni semantiche extra (pro capite / percentuali)
            norm_pop, as_percent = self._detect_population_normalization(user_request)
            if norm_pop is not None:
                params.normalize_by_population = norm_pop
                params.normalize_as_percent = as_percent
            # Metriche normalizzate verso nomi canonici
            params.metrics = self._normalize_metrics(params.metrics or [])
            # Regole aggiuntive per richieste su "redditi"
            self._apply_income_defaults(user_request, params)
            # Regole per richieste su popolazione/abitanti
            self._apply_population_defaults(user_request, params)
            return params
        except Exception as e:
            logger.error(f"LLM error in process_request: {e}")
            return self._fallback_parameters(user_request)

    def generate_commentary(self, df: pd.DataFrame, params: "QueryParameters") -> str:
        """
        Commento sintetico (3â€“6 bullet) sui dati passati.
        """
        try:
            df_safe = df.copy()
            if df_safe.shape[1] > 8:
                df_safe = df_safe.iloc[:, :8]
            if "anno" in df_safe.columns and df_safe["anno"].notna().any():
                try:
                    df_safe = df_safe.sort_values("anno").tail(50)
                except Exception:
                    df_safe = df_safe.tail(50)
            else:
                df_safe = df_safe.head(50)
            csv_preview = df_safe.to_csv(index=False)

            qtype = getattr(params.query_type, "value", params.query_type)
            metrics = ", ".join(params.metrics or [])
            comuni = ", ".join(params.comuni or []) if getattr(params, "comuni", None) else "(tutti i comuni)"
            if params.anno:
                period = str(params.anno)
            elif params.start_year and params.end_year:
                period = f"{params.start_year}-{params.end_year}"
            else:
                period = ""

            system = (
                "Sei un assistente di data analysis. Ricevi un CSV di preview (giÃ  aggregato). "
                "Rispondi con 3â€“6 punti elenco (ognuno inizia con '- ') in italiano: livelli, differenze comuni, trend, 1â€“2 insight. "
                "Niente premesse, niente codice, niente markdown extra."
            )
            user = (
                f"Contesto: tipo_query={qtype}; comuni={comuni}; metriche={metrics}; periodo={period}.\n"
                "CSV (prime/finali righe):\n" + csv_preview
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=300,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out.startswith("```"):
                out = out.strip("`")
            return out
        except Exception as e:
            logger.warning(f"Commentary fallback due to error: {e}")
            return "Osservazione sintetica non disponibile al momento. Prova a cambiare metrica o periodo."

    # ---------- Internals: Prompting ----------

    def _build_prompt(self, user_request: str) -> (str, str):
        variables: List[str] = []
        for _, r in self.variable_catalog.iterrows():
            v = str(r.get("variabile", "")).strip()
            lvl = str(r.get("livello", "")).lower()
            if v and not any(x in lvl for x in ["identificativo", "tecnico", "da escludere"]):
                variables.append(v)
        variables_list = "\t".join(sorted(set(variables))[:120])

        system = f"""
Sei un planner che converte richieste in linguaggio naturale in un JSON **valido** per una query dati.
Regole:
- Usa SOLO i nomi di colonne disponibili nel dataset.
- Imposta 'query_type' e 'chart_type' (preferenza: serie storiche â†’ chart=line).
- 'metrics' deve contenere nomi canonici (es: pop_totale, average_income, total_income, gini_index).
- Consenti 'derived_metrics' con campi: name, formula OPPURE (type in ['per_capita','share_of'] con metric/base o num/den).
- Se non Ã¨ specificato il periodo e NON Ã¨ una time series, l'anno sarÃ  deciso a valle (ultimo disponibile).
- Rispondi **solo** con JSON (senza testo extra).

Variabili disponibili:
{variables_list}

Sinonimi utili:
{self.catalog_text}

{self._examples_text()}
""".strip()

        user = f'Traduci in JSON la seguente richiesta: "{user_request.replace("\"", "\\\"")}".'
        return system, user

    def _call_llm(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        logger.debug(f"LLM raw: {raw[:500]}...")
        return raw

    # ---------- Internals: Parsing & Normalization ----------

    def _parse_llm_response(self, response: str) -> QueryParameters:
        t = (response or "").strip()

        if "```json" in t and "```" in t:
            t = t[t.find("```json") + 7 : t.rfind("```")]
        elif t.startswith("```") and t.endswith("```"):
            t = t.strip("`")
        if not t.strip().startswith("{"):
            i, j = t.find("{"), t.rfind("}")
            if i != -1 and j != -1 and j > i:
                t = t[i : j + 1]

        data = json.loads(t)

        qt = self._to_query_type(data.get("query_type", "single_comune"))
        chart = self._to_chart_type(data.get("chart_type", "bar"))
        metrics = self._normalize_metrics(data.get("metrics") or [])
        derived = data.get("derived_metrics") or []

        return QueryParameters(
            query_type=qt,
            chart_type=chart,
            comuni=data.get("comuni"),
            provincia=data.get("provincia"),
            regione=data.get("regione"),
            metrics=metrics or None,
            anno=data.get("anno"),
            start_year=data.get("start_year"),
            end_year=data.get("end_year"),
            n_results=data.get("n_results", 10),
            ascending=data.get("ascending", False),
            groupby=data.get("groupby"),
            derived_metrics=derived or None,
        )

    def _to_chart_type(self, value: Any) -> ChartType:
        if isinstance(value, ChartType):
            return value
        v = str(value).strip().lower()
        for ct in ChartType:
            if ct.value == v:
                return ct
        if v in ("linea", "linee"):
            return ChartType.LINE
        if v in ("barre", "colonne", "istogramma"):
            return ChartType.BAR
        return ChartType.BAR

    def _to_query_type(self, value: Any) -> QueryType:
        if isinstance(value, QueryType):
            return value
        v = str(value).strip().lower()
        for qt in QueryType:
            if qt.value == v:
                return qt
        aliases = {
            "serie storica": QueryType.TIME_SERIES,
            "confronto": QueryType.COMPARE_COMUNI,
            "confronto tra": QueryType.COMPARE_COMUNI,
            "singolo": QueryType.SINGLE_COMUNE,
            "classifica": QueryType.TOP_COMUNI,
            "statistiche": QueryType.STATISTICS,
            "reddito": QueryType.REDDITO_ANALYSIS,
            "provincia": QueryType.PROVINCIA_DATA,
            "regione": QueryType.REGIONE_DATA,
            "distribuzione": QueryType.DISTRIBUTION,
            "distribuzione redditi": QueryType.DISTRIBUTION,
        }
        if v in aliases:
            return aliases[v]
        if any(k in v for k in ["anni", "storico", "andamento", "nel tempo", "serie", "evoluzione", "trend", "tempo"]):
            return QueryType.TIME_SERIES
        if any(k in v for k in ["confronta", "compara", "paragona"]):
            return QueryType.COMPARE_COMUNI
        if "top" in v or "migliori" in v:
            return QueryType.TOP_COMUNI
        if "stat" in v:
            return QueryType.STATISTICS
        if "provinc" in v:
            return QueryType.PROVINCIA_DATA
        if "region" in v:
            return QueryType.REGIONE_DATA
        return QueryType.SINGLE_COMUNE

    def _normalize_metrics(self, metrics: List[Any]) -> List[str]:
        out = []
        known = list(self.metric_mapping.keys())
        import difflib
        for m in metrics:
            key = ""
            if isinstance(m, dict):
                key = (m.get("name") or m.get("metric") or m.get("id") or "").lower().strip()
            else:
                key = str(m).lower().strip()
            if not key:
                continue
            mapped = self.metric_mapping.get(key)
            if not mapped:
                alt = key.replace(" ", "_")
                mapped = self.metric_mapping.get(alt)
            if not mapped:
                close = difflib.get_close_matches(key, known, n=1, cutoff=0.75)
                if close:
                    mapped = self.metric_mapping.get(close[0], key)
            out.append(mapped or key)
        return out

    # ---------- Internals: Fallback & Heuristics ----------

    def _fallback_parameters(self, user_request: str) -> QueryParameters:
        text = (user_request or "").lower()
        if "distribuz" in text:
            qt, ch = QueryType.DISTRIBUTION, ChartType.BAR
        # Regola: 'redditi' generico -> time series dei redditi medi
        elif ("redditi" in text) and not any(q in text for q in [
            "totale", "imponibil", "pension", "dipendent", "autonom", "fabbricat",
            "imprenditor", "partecip", "addizional", "irpef", "complessiv",
        ]):
            qt, ch = QueryType.TIME_SERIES, ChartType.LINE
        elif any(k in text for k in ["confronta", "paragona", "compara", "comparare"]):
            qt, ch = QueryType.COMPARE_COMUNI, ChartType.BAR
        elif any(k in text for k in ["andamento", "anni", "serie", "trend", "nel tempo", "evoluzione", "storico", "tempo"]):
            qt, ch = QueryType.TIME_SERIES, ChartType.LINE
        elif "top" in text or "migliori" in text:
            qt, ch = QueryType.TOP_COMUNI, ChartType.BAR
        else:
            qt, ch = QueryType.SINGLE_COMUNE, ChartType.BAR

        if any(k in text for k in ["popolazione", "abitanti", "popola", "popolazion", "residenti"]):
            metrics = ["pop_totale"]
        elif ("ricchezza" in text) or ("reddito medio" in text):
            metrics = ["average_income"]
        elif "gini" in text:
            metrics = ["gini_index"]
        # Regola: redditi generici -> average_income
        elif "redditi" in text:
            metrics = ["average_income"]
        else:
            metrics = ["average_income"]

        norm_pop, as_percent = self._detect_population_normalization(text)
        return QueryParameters(
            query_type=qt,
            chart_type=ch,
            metrics=metrics,
            n_results=10,
            normalize_by_population=norm_pop,
            normalize_as_percent=as_percent,
        )

    def _detect_population_normalization(self, user_text: str):
        t = (user_text or "").lower()
        keywords_ratio = ["rispetto alla popolazione", "su popolazione", "per abitante", "pro capite", "per capita"]
        keywords_percent = ["percentuale", "percentuali", "quota", "%"]
        if any(k in t for k in keywords_percent):
            return True, True
        if any(k in t for k in keywords_ratio):
            return True, False
        return None, None

    def _apply_income_defaults(self, user_text: str, params: "QueryParameters") -> None:
        """Applica le regole richieste per 'redditi'.

        - Se vengono chiesti redditi in modo generico ("redditi" senza qualificatori),
          forza time_series + line e metrica average_income.
        - Se non vengono specificati quali redditi (metrics vuoto), imposta average_income.
        """
        t = (user_text or "").lower()
        # Qualificatori che rendono 'redditi' non generico
        qualifiers = [
            "totale", "imponibil", "pension", "dipendent", "autonom", "fabbricat",
            "imprenditor", "partecip", "addizional", "irpef", "complessiv",
        ]
        is_generic_redditi = ("redditi" in t) and not any(q in t for q in qualifiers)

        if is_generic_redditi:
            params.query_type = QueryType.TIME_SERIES
            params.chart_type = ChartType.LINE
            params.metrics = ["average_income"]
            return

        # Se non specificate metriche, default a average_income
        if not (params.metrics and len(params.metrics) > 0):
            params.metrics = ["average_income"]

        # Doctest-style example (informal):
        # >>> lp = LLMProcessor(api_key="test")  # doctest: +SKIP
        # >>> p = lp._fallback_parameters("Redditi Torino nel tempo")  # doctest: +SKIP
        # >>> lp._apply_income_defaults("Redditi Torino nel tempo", p)  # doctest: +SKIP
        # >>> p.query_type, p.chart_type, p.metrics  # doctest: +SKIP
        # (<QueryType.TIME_SERIES: 'time_series'>, <ChartType.LINE: 'line'>, ['average_income'])

    def _apply_population_defaults(self, text: str, params: "QueryParameters") -> "QueryParameters":
        t = (text or "").lower()
        if ("popolazione" in t) or ("abitanti" in t):
            params.metrics = ["pop_totale"]
            params.chart_type = ChartType.LINE
            params.query_type = QueryType.TIME_SERIES
        return params

# ---------- Internals: Catalog & Mappings ----------
    def _load_variable_catalog(self) -> pd.DataFrame:
        """Carica il catalogo variabili. Se non trova dizionario_variabili.csv,
        legge direttamente le colonne dal dataset principale (resources/df_ridotto_bot.csv o .xlsx)."""
        import os
        import pandas as pd

        cands = [
        os.getenv("VARIABLES_DICT", "").strip(),
        os.path.join(os.getcwd(), "resources", "dizionario_variabili.csv"),
        os.path.join(os.getcwd(), "dizionario_variabili.csv"),
        ]

        for p in cands:
            if p and os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                except Exception:
                    df = pd.read_csv(p, sep=';')
            df.columns = [c.strip().lower() for c in df.columns]
            for c in df.columns:
                if df[c].dtype == "object":
                    df[c] = df[c].astype(str).str.strip()
            print(f"âœ… Catalogo variabili caricato da file: {p}")
            return df

        # Fallback: genera il catalogo direttamente dal dataset principale
        dataset_candidates = [
        os.path.join("resources", "df_ridotto_bot.csv"),
        os.path.join("resources", "df_ridotto_bot.xlsx"),
        ]

        for d in dataset_candidates:
            if os.path.exists(d):
                try:
                    if d.endswith(".csv"):
                        cols = pd.read_csv(d, nrows=1).columns.tolist()
                    else:
                        cols = pd.read_excel(d, nrows=1).columns.tolist()
                    df = pd.DataFrame({
                        "variabile": cols,
                        "descrizione": ["" for _ in cols],
                        "sinonimi/termini collegati": ["" for _ in cols],
                        "livello": ["comunale" for _ in cols],
                    })
                    print(f"âœ… Catalogo variabili generato dal dataset principale ({len(cols)} colonne)")
                    return df
                except Exception as e:
                    print(f"âŒ Errore durante la lettura del dataset per il catalogo: {e}")

        # Fallback minimo (ultima spiaggia)
        df = pd.DataFrame([{
        "variabile": "pop_totale",
        "descrizione": "Popolazione residente",
        "sinonimi/termini collegati": "popolazione, abitanti, residenti",
        "livello": "comunale",
     }])
        print("âš ï¸ Nessun dizionario trovato, uso fallback minimo (solo pop_totale).")
        return df


    def _build_metric_mapping(self, catalog: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _, r in catalog.iterrows():
            lvl = str(r.get("livello", "")).lower()
            if "da escludere" in lvl:
                continue
            v = str(r.get("variabile", "")).strip()
            syn = str(r.get("sinonimi/termini collegati", ""))
            if not v:
                continue
            mapping[v.lower()] = v
            mapping[v.replace("_", " ").lower()] = v
            for tok in re.split(r"[;,/]\s*|\n", syn):
                t = tok.strip().lower()
                if t:
                    mapping[t] = v

        mapping.update(
            {
                "ricchezza": "average_income",
                "reddito medio": "average_income",
                "average income": "average_income",
                "abitanti": "pop_totale",
                "residenti": "pop_totale",
                "gini": "gini_index",
                "reddito totale": "total_income",
                "redditi": "average_income",
                "redditi medi": "average_income",
                "pensioni": "reddito_da_pensione_ammontare_in_euro",
                "pensionati": "reddito_da_pensione_frequenza",
                "dipendenti": "reddito_da_lavoro_dipendente_ammontare_in_euro",
                "imprese registrate": "imprese_registrate_prov",
                "imprese attive": "imprese_attive_prov",
                "iscrizioni imprese": "imprese_iscrizioni_prov",
                "cessazioni imprese": "imprese_cessazioni_prov",
                "saldo imprese": "imprese_saldo_prov",
                "brevetti": "brevetti_num_prov",
                "percentuale brevetti": "brevetti_pct_prov",
                "merci scaricate": "merci_scaricate_tonnellate",
                "anno di imposta": "anno_di_imposta",
                "capoluogo": "capoluogo_flag",
            }
        )
        return mapping

    def _catalog_preview(self, catalog: pd.DataFrame) -> str:
        def compact(row):
            return (
                f"- {row.get('variabile')}: {row.get('descrizione')} | "
                f"sinonimi: {row.get('sinonimi/termini collegati')} | livello: {row.get('livello')}"
            )
        return "\n".join([compact(r) for _, r in catalog.iterrows()])

    def _examples_text(self) -> str:
        return """
ESEMPI:
1. "Popolazione Bari e Napoli nel tempo":
{
  "query_type": "time_series", "chart_type": "line", "comuni": ["Bari", "Napoli"], "metrics": ["pop_totale"]
}
2. "Confrontami Milano, Roma, Padova e Bologna sulla ricchezza nel 2013":
{
  "query_type": "compare_comuni", "chart_type": "bar", "anno": 2013, "comuni": ["Milano", "Roma", "Padova", "Bologna"], "metrics": ["average_income"]
}
3. "Percentuale pensionati a Bari, Roma e Milano oggi":
{
  "query_type": "compare_comuni", "chart_type": "bar", "anno": 2023, "comuni": ["Bari", "Roma", "Milano"],
  "derived_metrics": [
    {"name": "percentuale_pensionati", "type": "share_of", "num": "reddito_da_pensione_ammontare_in_euro", "den": "pop_totale"}
  ]
}
        """.strip()
