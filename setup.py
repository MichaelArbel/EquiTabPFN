from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name="EquiTabPFN",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest", "black", "pre-commit"],
        "extra": [],
    },
)
