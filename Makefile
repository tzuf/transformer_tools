
.PHONY: clean-pyc clean-build

help:
	@echo "clean-pyc -- remove auxiliary python files"
	@echo "clean -- total cleaning of project files"
clean-pyc:
	find transformer_tools/ -name '*.pyc' -exec rm -f {} +
	find transformer_tools/ -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean: clean-pyc

test:
	#python -W ignore setup.py test
	#python -W ignore -m language_fragments.test.check_rule_lang
