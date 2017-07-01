# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
      setup_requires = f.readlines()

setup(name='rl-playground',
      version='0.1.0',
      description='Reinforcement Learning Playground',
      author='Pasquale Minervini',
      author_email='p.minervini@ucl.ac.uk',
      url='https://github.com/pminervini/rl-playground',
      test_suite='tests',
      license='MIT',
      install_requires=setup_requires,
      extras_require={
            'tests': ['pytest', 'pytest-pep8', 'pytest-xdist', 'pytest-cov'],
      },
      packages=find_packages())
