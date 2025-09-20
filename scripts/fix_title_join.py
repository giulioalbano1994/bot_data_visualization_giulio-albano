import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'main.py'
s = p.read_bytes()

# Replace any 'title = ... .join(title_parts)' with the canonical form
pattern = re.compile(br'title\s*=.*?\.join\(title_parts\)', re.S)
repl = b'title = " - ".join(title_parts)'
new = pattern.sub(repl, s)

p.write_bytes(new)
print('Fixed title join patterns:', s != new)
