```python
from setuptools import setup, find_packages

setup(
    name="crest",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "networkx>=2.6.2",
        "matplotlib>=3.4.3",
        "numpy>=1.21.2",
        "tqdm>=4.62.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="CREST: Causal Reasoning Enhancement through Structured Training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CREST",
)
