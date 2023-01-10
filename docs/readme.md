# Auto-Documentation

# Sphinx

All docstrings in microDL adhere to the sphinx format, 
and documentation can therefore be automatically generated with [Sphinx.](https://www.sphinx-doc.org/en/master/index.html)

Requirements for using the auto-documentation are sphinx and the theme sphinx_rtd_theme (installable with pip).

# Updating the Docs

For each release, go to docs/source/conf.py and update the release number.

If new modules have been added since the last build, you need to run the command
```buildoutcfg
sphinx-apidoc -o docs/source/ micro_dl/
```
from the main microDL directory to autogenerate the reStructuredText (rst) files.

A Makefile was already generated upon initialization of Sphinx (when running sphinx-quickstart), so the
next step is to run
```buildoutcfg
make -C docs/ html
```
Check for error messages and warning to make sure all docstring are correctly formatted.

Once updated docs are merged into main, the documentation can be hosted e.g. on
[GitHub Pages.](https://pages.github.com/)

There's a GitHub actions workflow (actions_docs.yml) that will 
automatically build the docs and push to the gh-pages branch when there's a push to the main branch.
The documentation will then be pushed to GitHub pages and the documentation is
published [here.](https://mehta-lab.github.io/microDL/)

Note: Python version is currently old because microDL version 1.0.0 still supports tensorflow.
Please update when updating to version 2.0.0.