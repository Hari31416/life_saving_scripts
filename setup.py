import life_saving_scripts as scripts
from setuptools import setup, find_packages

version = scripts.__version__

setup(
    name="life_saving_scripts",
    version=version,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "pillow",
        "matplotlib",
        "plotly",
    ],
    author="Harikesh Kushwaha",
    author_email="harikeshkumar0926@gmail.com",
    url="https://github.com/Hari31416/life_saving_scripts",
    keywords=[
        "personal",
        "scripts",
        "utilities",
    ],
    license="MIT",
    description="Some personal scripts that I use in my daily life.",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
