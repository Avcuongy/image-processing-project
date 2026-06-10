from setuptools import setup, find_packages

setup(
    name="image-processing-project",
    version="0.1.0",
    description="Image processing for face detection and recognition",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
)
