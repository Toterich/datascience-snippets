from setuptools import setup

setup(name='datascience-snippets',
      version='0.1',
      description='Collection of useful code snippets for plotting data, tuning model parameters and more in python.',
      url='https://github.com/Toterich/datascience-snippets.git',
      author='Toterich',
      author_email='mod4@gmx.de',
      license='MIT',
      packages=['datascience-snippets'],
      install_requires=['scikit-learn>=0.18', 'matplotlib', 'seaborn'],
      zip_safe=False)
