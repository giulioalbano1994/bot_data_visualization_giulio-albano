import io, os, re
from pathlib import Path
p = Path('modules/llm_processor.py')
text = p.read_text(encoding='utf-8')
# Normalize newlines
text = text.replace('\r\n', '\n')

# Fix docstring arrows line
text = text.replace(
    "Modulare: build_prompt ��' call_llm ��' parse_response ��' normalize/fallback.",
    "Modulare: build_prompt -> call_llm -> parse_response -> normalize/fallback."
)

# Fix commentary system text
text = text.replace(
    "Sei un assistente di data analysis. Ricevi un CSV di preview (gi�� aggregato). ",
    "Sei un assistente di data analysis. Ricevi un CSV di preview (gia' aggregato). "
)
text = text.replace(
    "Rispondi con 3�?"6 punti elenco (ognuno inizia con '- ') in italiano: livelli, differenze comuni, trend, 1�?"2 insight. ",
    "Rispondi con 3-6 punti elenco (ognuno inizia con '- ') in italiano: livelli, differenze tra comuni, trend, 1-2 insight. "
)

# Fix planner system text corrupted arrows/accents
text = text.replace(
    "preferenza: serie storiche ��' chart=line).",
    "preferenza: serie storiche -> chart=line)."
)
text = text.replace(
    "Se non �� specificato il periodo e NON �� una time series, l'anno sar�� deciso a valle (ultimo disponibile).",
    "Se non e' specificato il periodo e NON e' una time series, l'anno sara' deciso a valle (ultimo disponibile)."
)

# Fix try/except indentation in _load_variable_catalog
text = text.replace(
    "try:\n                    df = pd.read_csv(p)\n                    except Exception:\n                df = pd.read_csv(p, sep=\";\")",
    "try:\n                    df = pd.read_csv(p)\n                except Exception:\n                    df = pd.read_csv(p, sep=';')"
)

# Fix dataset candidates indentation (list items)
text = text.replace(
    "dataset_candidates = [\n        os.path.join(\"resources\", \"df_ridotto_bot.csv\"),\n        os.path.join(\"resources\", \"df_ridotto_bot.xlsx\"),\n        ]",
    "dataset_candidates = [\n            os.path.join(\"resources\", \"df_ridotto_bot.csv\"),\n            os.path.join(\"resources\", \"df_ridotto_bot.xlsx\"),\n        ]"
)

# Fix misplaced if indentation under for loop
text = text.replace(
    "for d in dataset_candidates:\n        if os.path.exists(d):",
    "for d in dataset_candidates:\n            if os.path.exists(d):"
)

# Sanitize corrupted print lines
lines = []
for line in text.split('\n'):
    stripped = line.strip()
    if 'Catalogo variabili caricato da file' in stripped:
        line = '                print(f"OK. Catalogo variabili caricato da file: {p}")'
    elif 'Catalogo variabili generato dal dataset principale' in stripped:
        line = '                    print(f"OK. Catalogo variabili generato dal dataset principale ({len(cols)} colonne)")'
    elif 'Errore durante la lettura del dataset per il catalogo' in stripped:
        line = '                except Exception as e:\n                    print(f"ERRORE durante la lettura del dataset per il catalogo: {e}")'
    elif 'Nessun dizionario trovato, uso fallback minimo' in stripped or 'Nessun dizionario trovato' in stripped:
        line = '        print("Nessun dizionario trovato, uso fallback minimo (solo pop_totale).")'
    lines.append(line)
text = '\n'.join(lines)

# Ensure section header is indented within class
text = text.replace('\n# ---------- Internals: Catalog & Mappings ----------', '\n    # ---------- Internals: Catalog & Mappings ----------')

# Also fix the odd bullet text in commentary docstring if any leftover corrupted markers
text = text.replace('3�?"6', '3-6').replace('1�?"2', '1-2')

# Write back with original newline style (CRLF)
Path('modules/llm_processor.py').write_text(text.replace('\n', '\r\n'), encoding='utf-8')
print('Done')
