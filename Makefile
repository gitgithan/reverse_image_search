default: pylint pytest black

pylint:
	find . -iname "*.py" ! -path "./venv/*" ! -path "./test*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -vv --color=yes

black:
	find . -iname "*.py" ! -path "./venv/*" ! -path "./test*" | xargs -n1 -I {}  black {}