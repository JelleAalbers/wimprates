try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()

setup(name='wimprates',
      version='0.1',
      description='Differential rates of WIMP-nucleus scattering',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/wimprates',
      license='MIT',
      py_modules=['wimprates'],
      install_requires=requirements,
      keywords='wimp,spin-dependent,spin-independent,bremsstrahlung',
      classifiers=['Intended Audience :: Developers',
                   'Development Status :: 3 - Alpha',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3'],
      zip_safe=False)
