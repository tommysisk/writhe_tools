import sys
from setuptools import setup, find_packages

# Explicit version check
if sys.version_info < (3, 7) or sys.version_info >= (3, 11):
    sys.exit("Error: Writhe_Package_Lite requires Python 3.7–3.10.")

setup(
    name='Writhe_Package_Lite',
    version='1',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    python_requires=">=3.7, <3.11",  # PEP 345 standard
    extras_require={
        'graph': [
            'torch-scatter>=2.1.1,<3.0',
            'pytorch_lightning>=2.0.9.post0,<3.0'
        ],
        'dev': [
            'pytest>=6.0.0,<8.0.0',
            'black>=22.0.0,<24.0.0',
            'flake8>=4.0.0,<6.0.0'
        ]
    },
    url='',
    license='MIT',
    author='Thomas Sisk',
    author_email='Thomas.r.Sisk.gr@dartmouth.edu',
    description='Compute writhe and train Writhe-PaiNN score based generative models with torch'
)