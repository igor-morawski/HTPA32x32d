from setuptools import setup
from setuptools import find_packages

long_description = '''
HTPA32x32d #TODO description
'''

setup(name='HTPA32x32d',
      version='1.1.1',
      description='Utilities to work with Heimann HTPA 32x32d using starter kit',
      long_description=long_description,
      author='Igor Morawski',
      url='https://github.com/igor-morawski/HTPA32x32d',
      license='MIT',
      install_requires=[
        "imageio>=2.8.0",
        "matplotlib>=3.2.1",
        "numpy>=1.18.4",
        "opencv-python>=4.2.0.34",
        "pandas>=1.0.3",
        "Pillow>=7.1.2",
        "scipy>=1.4.1",
      ],
      packages=find_packages())
