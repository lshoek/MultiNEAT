#!/usr/bin/env python
from skbuild import setup


setup(name='multineat',
      version='0.6',  # Update version in conda/meta.yaml as well
      packages=['multineat'],
      cmake_args=['-DGENERATE_PYTHON_BINDINGS:BOOL=ON'],
      install_requires=[
            'numpy',
      ],
      )
