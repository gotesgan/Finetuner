import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="finetuner",
    version="0.1.0",
    description="A command-line tool for fine-tuning language models on custom datasets",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/finetuner",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["transformers", "datasets", "unsloth", "trl", "torch"],
    entry_points={
        "console_scripts": [
            "finetuner=finetuner.cli:main",
        ],
    },
)
