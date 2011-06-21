from distutils.core import setup

setup(name='mtest',
      version='1.0',
      description='Two-sample test based on model selection and with better performance than the t-test on small sample sizes.',
      author='Pietro Berkes',
      author_email='pietro.berkes@googlemail.com',
      url='https://github.com/pberkes/mtest',
      license='LICENSE.txt',
      long_description=open('README.rst').read(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      requires=[
          'pymc',
          'scipy (>=0.8)',
          ],
      packages=['mtest'],
      package_data={'mtest': ['tables/*.npz']}
      )
