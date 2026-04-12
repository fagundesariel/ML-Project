# Paper Compilation (SBC Template)

This directory contains the LaTeX source files for the project paper using the SBC conference template.

---

# Project Structure

```text
paper/
├── main.tex
├── sbc-template.sty
├── sbc.bst
├── references.bib
├── figures/
└── sections/
```

* `main.tex` — Main LaTeX document
* `sections/` — Individual paper sections
* `figures/` — Images generated from notebooks
* `references.bib` — Bibliographic references

---

# Option 1 — Compile using Overleaf (Recommended)

This is the easiest and most reliable method.

## Steps:

1. Compress the `paper/` directory:

```bash
zip -r paper.zip paper/
```

2. Go to:

https://www.overleaf.com

3. Create a new project:

```
New Project → Upload Project
```

4. Upload:

```
paper.zip
```

5. Open:

```
main.tex
```

6. Click:

```
Recompile
```

---

# Option 2 — Compile locally (Linux / WSL)

## Install LaTeX dependencies

Run:

```bash
sudo apt update
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-lang-portuguese -y
```

---

## Compile the document

Navigate to the paper directory:

```bash
cd reports/paper
```

Run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This generates:

```text
main.pdf
```

---

# Notes

* Always run `bibtex` between LaTeX compilations to correctly generate references.
* Figures used in the paper should be placed inside:

```text
figures/
```

* Sections should be placed inside:

```text
sections/
```

and included in `main.tex` using:

```latex
\input{sections/section_name}
```
