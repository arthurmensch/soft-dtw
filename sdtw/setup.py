from numpy.distutils.core import setup, Extension

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('sdtw', parent_package, top_path)

    extensions = [Extension('sdtw.soft_dtw_fast',
                            sources=['sdtw/soft_dtw_fast.pyx'],
                            include_dirs=[numpy.get_include()])]

    config.ext_modules += extensions

    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
