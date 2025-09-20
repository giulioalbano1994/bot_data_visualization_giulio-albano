import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import pandas as pd


class ChartGenerator:
    def generate_chart(self, df: pd.DataFrame, chart_type: str, title: str, xlabel: str, ylabel: str) -> bytes:
        if df.empty:
            raise ValueError("DataFrame vuoto: impossibile generare il grafico.")

        # Determina colonna X: anno, comune, provincia, regione o prima colonna
        x_col = None
        for candidate in ['anno', 'comune', 'provincia', 'regione']:
            if candidate in df.columns:
                x_col = candidate
                break
        if not x_col:
            x_col = df.columns[0]

        y_cols = [col for col in df.columns if col != x_col]
        if not y_cols:
            raise ValueError("Nessuna colonna Y disponibile per il grafico.")

        # Rilevamento e ordinamento specifico per classi di reddito
        income_amount_cols = [
            'reddito_complessivo_da_0_a_10000_euro_ammontare_in_euro',
            'reddito_complessivo_da_10000_a_15000_euro_ammontare_in_euro',
            'reddito_complessivo_da_15000_a_26000_euro_ammontare_in_euro',
            'reddito_complessivo_da_26000_a_55000_euro_ammontare_in_euro',
            'reddito_complessivo_da_55000_a_75000_euro_ammontare_in_euro',
            'reddito_complessivo_da_75000_a_120000_euro_ammontare_in_euro',
            'reddito_complessivo_oltre_120000_euro_ammontare_in_euro',
        ]
        income_freq_cols = [
            'reddito_complessivo_da_0_a_10000_euro_frequenza',
            'reddito_complessivo_da_10000_a_15000_euro_frequenza',
            'reddito_complessivo_da_15000_a_26000_euro_frequenza',
            'reddito_complessivo_da_26000_a_55000_euro_frequenza',
            'reddito_complessivo_da_55000_a_75000_euro_frequenza',
            'reddito_complessivo_da_75000_a_120000_euro_frequenza',
            'reddito_complessivo_oltre_120000_euro_frequenza',
        ]
        income_label_map = {
            'reddito_complessivo_da_0_a_10000_euro_ammontare_in_euro': 'Da 0 a 10 mila',
            'reddito_complessivo_da_10000_a_15000_euro_ammontare_in_euro': 'Da 10 mila a 15 mila',
            'reddito_complessivo_da_15000_a_26000_euro_ammontare_in_euro': 'Da 15 mila a 26 mila',
            'reddito_complessivo_da_26000_a_55000_euro_ammontare_in_euro': 'Da 26 mila a 55 mila',
            'reddito_complessivo_da_55000_a_75000_euro_ammontare_in_euro': 'Da 55 mila a 75 mila',
            'reddito_complessivo_da_75000_a_120000_euro_ammontare_in_euro': 'Da 75 mila a 120 mila',
            'reddito_complessivo_oltre_120000_euro_ammontare_in_euro': 'Oltre 120 mila',
            'reddito_complessivo_da_0_a_10000_euro_frequenza': 'Da 0 a 10 mila',
            'reddito_complessivo_da_10000_a_15000_euro_frequenza': 'Da 10 mila a 15 mila',
            'reddito_complessivo_da_15000_a_26000_euro_frequenza': 'Da 15 mila a 26 mila',
            'reddito_complessivo_da_26000_a_55000_euro_frequenza': 'Da 26 mila a 55 mila',
            'reddito_complessivo_da_55000_a_75000_euro_frequenza': 'Da 55 mila a 75 mila',
            'reddito_complessivo_da_75000_a_120000_euro_frequenza': 'Da 75 mila a 120 mila',
            'reddito_complessivo_oltre_120000_euro_frequenza': 'Oltre 120 mila',
        }

        # Se tutte (o alcune) le colonne Y appartengono alle classi di reddito, ordina secondo la sequenza desiderata
        def _ordered_subset(order_list, items):
            s = set(items)
            return [c for c in order_list if c in s]

        is_income_amount = any(c in income_amount_cols for c in y_cols)
        is_income_freq = any(c in income_freq_cols for c in y_cols)
        if is_income_amount:
            y_cols = _ordered_subset(income_amount_cols, y_cols)
        elif is_income_freq:
            y_cols = _ordered_subset(income_freq_cols, y_cols)

        df_plot = df.set_index(x_col)[y_cols]
        # Dimensioni dinamiche in base ai punti e alle serie
        n_points = len(df_plot)
        n_series = len(y_cols)
        width = min(18, 6 + 0.25 * max(n_points, 8))
        height = 4.5 + 0.3 * min(n_series, 8)
        fig, ax = plt.subplots(figsize=(width, height))

        # Selezione tipo di grafico
        chart_type = chart_type.lower()
        if chart_type == "bar":
            df_plot.plot(kind="bar", ax=ax)
        elif chart_type == "pie":
            if len(df_plot.columns) != 1:
                raise ValueError("Il grafico a torta richiede una sola metrica.")
            df_plot.iloc[:, 0].plot(kind="pie", ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
        elif chart_type == "scatter":
            for col in y_cols:
                ax.scatter(df[x_col], df[col], label=col)
        elif chart_type == "box":
            df[y_cols].plot(kind="box", ax=ax)
        elif chart_type == "histogram":
            df[y_cols].plot(kind="hist", ax=ax, alpha=0.7, bins=15)
        else:  # default: line
            df_plot.plot(ax=ax)

        # Impostazioni grafiche
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # Gestione legenda: per singola serie non mostrare; per classi reddito semplificare etichette e titolo
        if chart_type != "pie":
            show_legend = len(y_cols) > 1
            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                # Se sono classi di reddito, mappa le etichette a valori compatti e aggiungi titolo
                if (is_income_amount or is_income_freq) and labels:
                    labels = [income_label_map.get(l, l) for l in labels]
                    ax.legend(handles, labels, loc="best", title="fascia di reddito")
                else:
                    ax.legend(loc="best")
            else:
                # Nessuna legenda per grafici a singola serie (es. classe/valore)
                pass
        # Ruota le etichette dell'asse X se affollate o se sono comuni
        if n_points > 8 or x_col in ("comune", "provincia", "regione"):
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

        # Se sembra una percentuale, applica formattazione dell'asse Y a step regolari (5%)
        is_percent = any(str(c).endswith('_pct_pop') for c in y_cols) or ('%' in str(ylabel))
        if is_percent:
            try:
                y_max = float(pd.to_numeric(df_plot.max().max(), errors='coerce'))
                upper = min(100.0, ((int(y_max) // 5) + 2) * 5)
                ax.set_ylim(0, upper)
            except Exception:
                pass
            ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

        # Salva immagine in memoria
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()
