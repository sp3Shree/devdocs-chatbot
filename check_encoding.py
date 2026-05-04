import os

SKIP = {'.venv', '.git', '__pycache__', 'build', 'dist', 'node_modules', 'data'}
bad = []

for root, dirs, files in os.walk('.'):
    if any(part in SKIP for part in root.replace('\\','/').split('/')):
        continue
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            with open(p, 'rb') as fh:
                b = fh.read()
            try:
                b.decode('utf-8')
            except UnicodeDecodeError:
                bad.append(p)

print("\n".join(bad))
