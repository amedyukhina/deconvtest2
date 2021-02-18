from setuptools import setup

setup(
    name='deconvtest2_modules',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['deconvtest2_modules',
              'deconvtest2_modules.ground_truth',
              'deconvtest2_modules.psf',
              'deconvtest2_modules.transforms',
              'deconvtest2_modules.deconvolution',
              'deconvtest2_modules.evaluation'],
    license='BSD-3-Clause',
    include_package_data=True,

    test_suite='deconvtest2_modules.tests',

    install_requires=[
        'numpy',
        'scipy',
        'ddt',
    ],
)
