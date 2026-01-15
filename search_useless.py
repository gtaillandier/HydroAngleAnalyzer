import os
import re

patterns = [
    re.compile(r"(\w+)\s*=\s*\1(\s|#|$)"),
    re.compile(r"import\s+(\w+)\s+as\s+\1(\s|$)"),
    re.compile(r"from\s+[\w\.]+\s+import\s+(\w+)\s+as\s+\1(\s|$)"),
    re.compile(r"self\.(\w+)\s*=\s*self\.\1(\s|$)"),
]

root_dir = (
    "/home/gtaillandier/Documents/project_with_edo/"
    "edocolad/lib_python/HydroAngleAnalyzer"
)

for root, dirs, files in os.walk(root_dir):
    if ".git" in dirs:
        dirs.remove(".git")
    if ".venv" in dirs:
        dirs.remove(".venv")
    if "__pycache__" in dirs:
        dirs.remove("__pycache__")

    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    for pattern in patterns:
                        if pattern.search(line):
                            print(f"{path}:{i}: {line.strip()}")
