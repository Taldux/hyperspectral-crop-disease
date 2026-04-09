from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = 'hyperspectral-crop-disease'
copyright = '2026, Taldux'
author = 'Taldux'

version = '0.1.0'
release = '0.1.0'

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'
autoclass_content = 'class'
autodoc_typehints = 'description'

html_title = 'hyperspectral-crop-disease documentation'

html_theme = 'alabaster'
html_static_path = ['_static']
