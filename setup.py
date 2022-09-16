#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='mop',
      version='0.0.1',
      description='Motion Prior with global motion',
      author='Nitin Saini',
      author_email='nitin.ppnp@gmail.com',
      url='',
      install_requires=[
            'torch'
      ],
      packages=find_packages('src'),
      package_dir={'':'src'},
      )