from setuptools import setup

setup(
    name='deconvtest',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['deconvtest',
              'deconvtest.methods',
              'deconvtest.methods.ground_truth',
              'deconvtest.methods.psf',
              'deconvtest.methods.convolution',
              'deconvtest.methods.transforms',
              'deconvtest.methods.deconvolution',
              'deconvtest.methods.evaluation',
              'deconvtest.methods.datagen',
              'deconvtest.methods.training',
              'deconvtest.core',
              'deconvtest.core.shapes',
              'deconvtest.core.utils',
              'deconvtest.framework',
              'deconvtest.framework.module',
              'deconvtest.framework.workflow'
              ],
    license='BSD-3-Clause',
    include_package_data=True,

    test_suite='deconvtest.tests',

    install_requires=[
        'numpy',
        'scipy',
        'ddt',
        'pytest',
        'tqdm',
        'scikit-image',
        'pandas',
        'am_utils',
        'csbdeep',
        'matplotlib'
    ],
    dependency_links=[
        "https://github.com/amedyukhina/am_utils/releases/",
    ],
)
