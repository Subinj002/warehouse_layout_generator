from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="warehouse-layout-generator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for generating and optimizing warehouse layouts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/warehouse-layout-generator",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "ezdxf>=0.17.0",
        "pydantic>=1.8.0",
        "click>=8.0.0",
        "PyQt5>=5.15.0;platform_system!='Darwin' or python_version<'3.11'",
        "PyQt6>=6.2.0;platform_system=='Darwin' and python_version>='3.11'",
        "scikit-learn>=1.0.0",
        "jsonschema>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "isort>=5.0.0",
            "pytest-cov>=2.0.0",
        ],
        "ai": [
            "tensorflow>=2.8.0",
            "torch>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "warehouse-generator=warehouse_layout_generator.main:cli",
        ],
    },
)