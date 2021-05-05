try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import damage 

SHORT = 'damage'

setup(
    name='damage',
    version=damage.__version__,
    packages=[
        'damage'
    ],
    url='https://github.com/javiermas/building_damage/',
    classifiers=(
        'Programming Language :: Python :: 3.6'
    ),
    description=SHORT,
    long_description=SHORT,
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
