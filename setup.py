from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1.0a",
    packages=find_packages(where="nn_x"),
    package_dir={"": "nn_x"},
    include_package_data=True,
    # No install_requires since dependencies are managed separately
    # You can add metadata below as needed:
    # author="Your Name",
    # author_email="your.email@example.com",
    # description="A short description of your package",
    # url="https://github.com/yourusername/your-repo",
)
