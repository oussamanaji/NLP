from setuptools import setup, find_packages

setup(
    name="biasguard",
    version="0.1.0",
    author="Mohamed Oussama Naji",
    author_email="mohamedoussama.naji@georgebrown.ca",
    description="BiasGuard: Advanced Bias Mitigation in Large Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/oussamanaji/NLP/tree/main/biasguard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    entry_points={
        "console_scripts": [
            "biasguard=experiments.run_experiment:main",
        ],
    },
)
