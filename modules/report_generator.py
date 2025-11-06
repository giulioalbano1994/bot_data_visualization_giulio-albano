import io
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .llm_processor import QueryParameters, QueryType, ChartType


class ReportGenerator:
    """
    Crea un PDF con 3 grafici chiave per un comune:
    - Popolazione (serie)
    - Reddito medio (serie)
    - Gini index (serie)
    """

    def __init__(self, chart_generator, map_generator):
        self.chart_generator = chart_generator
        self.map_generator = map_generator

    def build_city_report(self, comune: str, df_manager) -> Tuple[bytes, str]:
        # Prepara tre query
        q1 = QueryParameters(query_type=QueryType.TIME_SERIES, chart_type=ChartType.LINE, comuni=[comune], metrics=["pop_totale"])
        q2 = QueryParameters(query_type=QueryType.TIME_SERIES, chart_type=ChartType.LINE, comuni=[comune], metrics=["average_income"])
        q3 = QueryParameters(query_type=QueryType.TIME_SERIES, chart_type=ChartType.LINE, comuni=[comune], metrics=["gini_index"])

        charts = []
        titles = ["Popolazione", "Reddito medio", "Gini index"]

        for qp, title in zip([q1, q2, q3], titles):
            df, xlabel, ylabel, meta = df_manager.query_data(qp)
            subtitle = self._subtitle_from_meta(meta)
            img = self.chart_generator.generate_chart(
                df,
                chart_type=qp.chart_type.value,
                title=title,
                subtitle=subtitle,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            charts.append(img)

        # PDF
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        margin = 1.5 * cm
        c.setTitle(f"Report socio-economico - {comune}")

        # Cover
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, h - margin - 10, f"Report socio-economico - {comune}")
        c.setFont("Helvetica", 10)
        c.drawString(margin, h - margin - 28, f"Generato il {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.drawString(margin, h - margin - 42, "Indicatori chiave: Popolazione, Reddito medio, Gini index")
        c.showPage()

        # 3 pagine con grafici
        for i, (img_bytes, title) in enumerate(zip(charts, titles), start=1):
            c.setFont("Helvetica-Bold", 13)
            c.drawString(margin, h - margin - 10, f"{i}. {title}")
            # Inserisci immagine
            img_buf = io.BytesIO(img_bytes)
            # Posizionamento
            img_w = w - 2 * margin
            img_h = h - 3 * margin
            c.drawImage(ImageReader(img_buf), margin, margin, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
            c.showPage()

        c.save()
        pdf_bytes = buf.getvalue()
        buf.close()
        filename = f"report_{comune.replace(' ', '_').lower()}.pdf"
        return pdf_bytes, filename

    def _subtitle_from_meta(self, meta: dict) -> str:
        src = ", ".join(sorted(meta.get("sources", []))) if meta else ""
        latest = meta.get("latest_year") if meta else None
        coverage = meta.get("coverage_str") if meta else ""
        bits = []
        if src:
            bits.append(f"Fonte: {src}")
        if latest:
            bits.append(f"Ultimo anno: {latest}")
        if coverage:
            bits.append(f"Coverage: {coverage}")
        return " | ".join(bits)

