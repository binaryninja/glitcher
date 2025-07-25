import os
from setuptools import setup, find_packages

setup(
    name="glitcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core ML dependencies
        "torch==2.7.1",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",

        # Core utility dependencies
        "tqdm>=4.64.0",
        "numpy>=1.21.0",

        # Data handling
        "pandas>=1.5.0",

        # Visualization (required for GUI features)
        "matplotlib>=3.5.0",

        # API providers
        "mistralai>=1.0.0",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "requests>=2.25.0",

        # Configuration and environment
        "python-dotenv>=0.19.0",

        # Quantization support
        "bitsandbytes>=0.41.0",
    ],
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],

        # Enhanced visualization
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.17.0",
        ],

        # Web interface
        "web": [
            "flask>=2.3.0",
            "plotly>=5.17.0",
            "jinja2>=3.1.0",
            "werkzeug>=2.3.0",
            "gunicorn>=21.2.0",
        ],

        # Enhanced CLI
        "cli": [
            "rich>=12.0.0",
        ],

        # Advanced logging
        "logging": [
            "structlog>=22.0.0",
        ],

        # Configuration management
        "config": [
            "pyyaml>=6.0",
        ],

        # Async support
        "async": [
            "aiohttp>=3.8.0",
        ],

        # Complete installation
        "all": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "seaborn>=0.11.0",
            "plotly>=5.17.0",
            "flask>=2.3.0",
            "jinja2>=3.1.0",
            "werkzeug>=2.3.0",
            "gunicorn>=21.2.0",
            "rich>=12.0.0",
            "structlog>=22.0.0",
            "pyyaml>=6.0",
            "aiohttp>=3.8.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'glitcher=glitcher.cli:main',
            'glitch-scan=glitcher.scan_and_validate:main',
            'glitch-classify=glitcher.classify_glitches:main',
        ],
    },
    author="Binary Ninja",
    description="A CLI tool for mining and testing glitch tokens in language models",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/binaryninja/glitcher",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    keywords="machine-learning nlp security prompt-injection glitch-tokens language-models",
    project_urls={
        "Bug Reports": "https://github.com/binaryninja/glitcher/issues",
        "Source": "https://github.com/binaryninja/glitcher",
        "Documentation": "https://github.com/binaryninja/glitcher/blob/main/README.md",
    },
)
