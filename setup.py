#!/usr/bin/env python3
"""
Pocket Vector Database - Setup Script
A lightweight, fast, offline-ready vector database for Python and mobile/edge environments
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="pocketvectordb",
    version="1.0.0",
    description="A lightweight, fast, offline-ready vector database for Python and mobile/edge environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ThatFkrDurk561",
    author_email="ddurkee651@gmail.com",
    url="",
    py_modules=["pocketvectordb"],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="vector database semantic search similarity search embeddings",
)
