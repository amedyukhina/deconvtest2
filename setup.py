from setuptools import setup

setup(
    name='deconvtest',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['deconvtest',
              'deconvtest.modules',
              'deconvtest.modules.ground_truth',
              'deconvtest.modules.psf',
              'deconvtest.modules.transforms',
              'deconvtest.modules.deconvolution',
              'deconvtest.modules.evaluation',
              'deconvtest.core',
              'deconvtest.core.shapes',
              'deconvtest.core.utils',
              'deconvtest.framework',
              'deconvtest.framework.module',
              'deconvtest.framework.step',
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
        'scikit-image'
    ],
)
