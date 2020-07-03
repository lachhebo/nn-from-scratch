import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn_from_scratch",
    version="0.0.1",
    author="Leo & Isma ",
    description="An nn_from_scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lachhebo/pyclustertend",
    keyword=["ml"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)