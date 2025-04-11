# setup.py
from setuptools import setup, find_packages

setup(
    name="emu",
    version="3.5.2",
    description="Species-level taxonomic abundance for full-length 16S reads",
    author="Emu Contributors",
    author_email="your.email@example.com",  # Update this
    url="https://github.com/treangenlab/emu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "emu=emu.cli:main",
        ],
    },
    install_requires=[
        "numpy>=1.11",
        "pandas>=1.1",
        "pysam>=0.15",
        "biopython",
        "flatten-dict",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)