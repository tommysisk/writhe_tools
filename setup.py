from setuptools import setup, find_packages
import os

# grab requirements from requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='Writhe_Package_Lite',
    version='1',
    packages=find_packages(),
    install_requires=install_requires,
    url='',
    license='',
    author='Thomas Sisk',
    author_email='Thomas.r.Sisk.gr@dartmouth.edu',
    description='Compute writhe and train Writhe-PaiNN score based generative models with torch'
    )
