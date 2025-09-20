from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'main.py'
s = p.read_text(encoding='utf-8', errors='replace')
s = s.replace('title = " - ".join(title_parts)', 'title = " - ".join([p for p in parts if p])')
p.write_text(s, encoding='utf-8')
print('Patched plot title join if needed')
