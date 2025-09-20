import io
path = 'main.py'
s = io.open(path,'r',encoding='utf-8',errors='replace').read().splitlines()
for i in range(320, 371):
    if i-1 < len(s):
        line = s[i-1]
        tag = ''
        ls = line.strip()
        if ls.startswith('try:'):
            tag = ' <-- TRY'
        elif ls.startswith('except'):
            tag = ' <-- EXCEPT'
        print(f"{i:04d}: {line}{tag}")
