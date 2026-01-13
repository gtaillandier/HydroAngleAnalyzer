---
description: How to rebuild the Sphinx documentation
---

To rebuild the documentation, you should run the following commands from the project root:

1.  **Navigate to the documentation directory**:
    ```bash
    cd docs
    ```

2.  **Clean the previous build (recommended)**:
    ```bash
    make clean
    ```

3.  **Build the HTML documentation**:
    ```bash
    make html
    ```

The generated documentation will be available at `docs/build/html/index.html`.

### Troubleshooting
If you see errors about missing modules, ensure you have the package installed in editable mode:
```bash
pip install -e .
```

If you want to force a full rebuild of all pages:
```bash
cd docs
rm -rf build/
make html
```
