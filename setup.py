import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gapfill",
    version="0.0.6",
    author="Moon Ki Jung",
    author_email="m.k.jung@outlook.com",
    description="GapFill: A Python module of gap filling functions for motion capture marker data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkjung99/gapfill",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)