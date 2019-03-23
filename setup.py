import setuptools

readme = open('README.md').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()

setuptools.setup(
    name='wimprates',
    version='0.2.0',
    description='Differential rates of WIMP-nucleus scattering',
    long_description=readme + '\n\n' + history,
    author='Jelle Aalbers',
    url='https://github.com/jelleaalbers/wimprates',
    license='MIT',
    packages=setuptools.find_packages(),
    setup_requires=['pytest-runner'],
    install_requires=requirements,
    tests_require=requirements + ['pytest'],
    keywords='wimp,spin-dependent,spin-independent,bremsstrahlung,migdal',
    classifiers=['Intended Audience :: Science/Research',
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3'],
    zip_safe=False)
