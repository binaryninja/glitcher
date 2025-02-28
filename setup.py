from setuptools import setup, find_packages

setup(
    name="glitcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "tqdm>=4.64.0",
    ],
    entry_points={
        'console_scripts': [
            'glitcher=glitcher.cli:main',
            'glitch-scan=glitcher.scan_and_validate:main',
            'glitch-classify=glitcher.classify_glitches:main',
        ],
    },
    author="Binary Ninja",
    description="A CLI tool for mining and testing glitch tokens in language models",
    url="https://github.com/binaryninja/glitcher",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)