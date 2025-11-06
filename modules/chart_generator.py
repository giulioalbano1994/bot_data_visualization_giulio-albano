import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import pandas as pd

# Palette color-blind friendly
PALETTE = None  # we'll rely on matplotlib default 'tab10' which is fine


class ChartGenerator:
    def __init__(self):
        # Global style tweaks
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "font.size": 11,
        })

    def generate_chart(self, df: pd.DataFrame, chart_type: str, title: str, xlabel: str, ylabel: str, subtitle: str = "") -> bytes:
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

        df_plot = df.set_index(x_col)[y_cols]

        # Dimensioni dinamiche
        n_points = len(df_plot)
        n_series = len(y_cols)
        width = min(18, 6 + 0.25 * max(n_points, 8))
        height = 4.8 + 0.32 * min(n_series, 8)
        fig, ax = plt.subplots(figsize=(width, height))

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

        # Titoli
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if subtitle:
            # Usa spazio sotto al titolo principale
            plt.suptitle(subtitle, y=0.97, fontsize=9, color="#444")
        # Legenda smart
        if chart_type != "pie":
            if len(y_cols) > 1:
                ax.legend(loc="best")
            else:
                ax.legend().remove()

        # X crowded?
        if n_points > 8 or x_col in ("comune", "provincia", "regione"):
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

        # Percent formatter
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

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close()
        return buf.getvalue()
