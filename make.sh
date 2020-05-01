#!/bin/sh
rm dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
