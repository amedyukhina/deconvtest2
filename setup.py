from setuptools import setup

setup(
    name='deconvtest2_core',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['deconvtest2_core',
              'deconvtest2_core.shapes',],
    license='BSD-3-Clause',
    include_package_data=True,

    test_suite='deconvtest2_core.tests',

    install_requires=[
        'numpy',
        'scipy',
        'ddt',
    ],
)
