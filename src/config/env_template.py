from pathlib import Path
import re

# Path to your .env file
env_path = Path(".env")
template_path = Path(".env_template")

if not env_path.exists():
    print(".env not found.")
    exit(1)

lines = env_path.read_text(encoding="utf-8").splitlines()
out_lines = []

for line in lines:
    # Preserve comments and empty lines
    if line.strip().startswith("#") or not line.strip():
        out_lines.append(line)
        continue

    # Extract key=value
    match = re.match(r'\s*([^#=\s]+)\s*=\s*["\']?(.*?)["\']?\s*(#.*)?$', line)
    if match:
        key = match.group(1)
        # Replace value with empty quotes
        out_lines.append(f'{key} = ""')
    else:
        out_lines.append(line)

template_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"✅ Generated {template_path}")
