from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [] 
# Include any requirements needed to use your package. You don't need anything that comes as part of the standard Python library

setup(
    name="synthetic-data-metrics",
    version="0.0.1",
    author="A. Smith",
    author_email="asmith@gmail.com",
    description="A package to do something",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/your_package/homepage/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)