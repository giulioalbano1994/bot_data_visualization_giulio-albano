import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'main.py'
s = p.read_text(encoding='utf-8', errors='replace')
new = re.sub(r'title_dist\s*=\s*f"Distribuzione dei redditi.*?\{comuni_label\}"',
             'title_dist = f"Distribuzione dei redditi - {comuni_label}"', s)
p.write_text(new, encoding='utf-8')
print('Fixed title_dist line:', s != new)
