from distutils.core import setup

REQUIRED_PACKAGES = [
    'pandas'
]

EXTRA_PACKAGES = {
    'test': [
        'pytest',
    ]
}

setup(
    name='famfaceangles',
    version='0.0.1dev',
    license='Apache License 2.0',
    author='Matteo Visconti di Oleggio Castello',
    author_email='matteo.visconti@berkeley.edu',
    description='Analysis code and script for famface angles',
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
)
