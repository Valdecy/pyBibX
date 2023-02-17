from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pyBibX',
    version='1.5.4',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyBibX',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'bertopic',
        'bert-extractive-summarizer',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'plotly',
        'sentencepiece',
        'squarify',
        'sklearn',
        'torch', 
        'torchvision',
        'torchaudio',
        'transformers',
        'umap-learn',
        'wordcloud'
    ],
    zip_safe=True,
    description='A Bibliometric and Scientometric Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
