#!/usr/bin/env python

from distutils.core import setup
import glob
import os

setup(name='gpt',
      version='1.0',
      # description='Python Distribution Utilities',
      # author='Greg Ward',
      # author_email='gward@python.net',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages=['.', 'dataloaders', 'models', 'scripts', 'utils'],
      package_data = {
                'dataloaders': ['*.json'],
            },
     )
