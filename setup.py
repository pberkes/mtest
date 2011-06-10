from distutils.core import setup

# TODO: dependencies for PYPI

setup(name='mtest',
      version='1.0',
      description='Two-sample test based on model selection and with better performance than the t-test on small sample sizes.',
      author='Pietro Berkes',
      author_email='pietro.berkes@googlemail.com',
      url='https://github.com/pberkes/mtest',
      license='LICENSE.txt',
      long_description=open('README.rst').read(),
      packages=['mtest', 'mtest.test']
      )
