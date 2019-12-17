import os
# Setuptools should be preferred:
# * https://stackoverflow.com/a/25372045/5578392
# * https://stackoverflow.com/a/14753678/5578392
#
# https://stackoverflow.com/a/41573588/5578392
from setuptools import setup
from setuptools import find_packages

with open(os.path.join('./', 'VERSION')) as version_file:
    project_version = version_file.read().strip()

setup(
    name='pypaddle',
    version=project_version,
    description='Innvariant Group Sparse Neural Network Tools',
    author='Chair of Data Science, University of Passau',
    author_email='julian.stier@uni-passau.de',
    url='https://github.com/innvariant/pypaddle',
    download_url = 'https://github.com/innvariant/pypaddle/archive/%s.tar.gz' % project_version,
    keywords=['sparse', 'neural', 'networks', 'graph theory', 'pytorch'],
    packages=find_packages(),
    license="GPL3 License",
    install_requires=['pip', 'setuptools>=18.0', 'torch>=1.3', 'networkx'],
    dependency_links=[],
    classifiers = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)