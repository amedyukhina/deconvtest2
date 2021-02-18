from setuptools import setup

setup(
    name='deconvtest2',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['deconvtest2',
              'deconvtest2.modules',
              'deconvtest2.modules.ground_truth',
              'deconvtest2.modules.psf',
              'deconvtest2.modules.transforms',
              'deconvtest2.modules.deconvolution',
              'deconvtest2.modules.evaluation',
              'deconvtest2.core',
              'deconvtest2.core.shapes',
              'deconvtest2.core.utils'
              ],
    license='BSD-3-Clause',
    include_package_data=True,

    test_suite='deconvtest2.tests',

    install_requires=[
        'numpy',
        'scipy',
        'ddt',
    ],
)
