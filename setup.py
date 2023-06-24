from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='traffic_signs_code',
      version="0.0.1",
      description="Traffic Sign detection ",
      author="Madhu, James, Mohamed, Ludwig",
      author_email="ludwigmatuschka@gmail.com",
      contrib_email= "mohamedscience7@gmail.com",
      #url="https://github.com/lewagon/taxi-fare",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
