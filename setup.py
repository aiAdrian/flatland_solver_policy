# python setup.py sdist
# pip install twine
# python setup.py bdist_wheel
# python setup.py sdist
# twine upload dist/*

from setuptools import find_packages
from setuptools import setup

setup(
    name='flatland-solver-policy',
    version='0.0.1',
    author='Adrian Egli',
    author_email="3dhelp@gmail.com",
    description='Flatland solvers',
    url='https://github.com/aiAdrian/flatland-solver-policy',
    keywords='flatland, railway, extension, dynamics, simulation, multi-agent, reinforcement learning',
    python_requires='>=3.6, <4',
    packages=find_packages('.'),
    install_requires=[
        'flatland-railway-extension',
        'torch',
        'tensorboard',
        'gym',
        'numpy'
    ],
)