from setuptools import setup, find_packages

setup(
    name="plant_disease_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.17.1',
        'numpy==1.24.3',
        'pandas==2.0.3',
        'Pillow==10.0.0',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.2'
    ],
)
