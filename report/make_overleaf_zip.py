"""Package the LaTeX report as a zip that can be uploaded to Overleaf (New Project -> Upload Project).

Run from the repository root:  python report/make_overleaf_zip.py
The project is standalone: it contains main.tex, preamble.tex (unchanged from the organizers' template),
main.bib and the two figures. Compiled and page-checked with tectonic (see README).
"""
import os
import zipfile

SRC = 'report/overleaf'
FILES = ['main.tex', 'preamble.tex', 'main.bib', 'fig_cases.png', 'fig_uncertainty.pdf']
out = os.path.join(SRC, 'overleaf_project.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.join(SRC, f), f)
print('wrote', out, os.path.getsize(out), 'bytes')
