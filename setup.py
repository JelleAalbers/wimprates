import setuptools

readme = open('README.md').read()
history = open('HISTORY.md').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()

setuptools.setup(
    name='wimprates',
    version='0.4.0',
    description='Differential rates of WIMP-nucleus scattering',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author='Jelle Aalbers',
    url='https://github.com/jelleaalbers/wimprates',
    license='MIT',
    packages=setuptools.find_packages(),
    setup_requires=['pytest-runner'],
    install_requires=requirements,
    package_dir={'wimprates': 'wimprates'},
    package_data={'wimprates': [
        'data/bs/*', 'data/migdal/*', 'data/sd/*', 'data/dme/*']},
    tests_require=requirements + ['pytest'],
    keywords='wimp,spin-dependent,spin-independent,bremsstrahlung,migdal',
    classifiers=['Intended Audience :: Science/Research',
                 'Development Status :: 3 - Alpha',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3'],
    zip_safe=False)
