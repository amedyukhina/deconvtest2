from setuptools import setup

setup(
    name='deconvtest2_modules',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['deconvtest2_modules'],
    license='BSD-3-Clause',
    include_package_data=True,

    test_suite='deconvtest2_modules.tests',

    install_requires=[
        'numpy',
        'scipy',
        'ddt',
    ],
)
