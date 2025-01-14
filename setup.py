from setuptools import setup, find_packages

NAME ='wassa'
import wassa
VERSION = "1.0"

setup(
    name=NAME,
    version=VERSION,
    package_dir={'hots': NAME},
    packages=find_packages(),
    author='Antoine Grimaldi, CerCo (CNRS)',
    author_email='antoine.grimaldi@cnrs.fr',
    url = 'https://github.com/AntoineGrimaldi/wassa',
    description=' This is a collection of python scripts to extract spiking motifs in raster plots',
    long_description=open('README.md').read(),
    license='GNU General Public License v3.0',
)