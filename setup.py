from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]


setup(
    name='heckles-stats',
    packages=['heckles_stats'],
    scripts=['bin/heckles-stats'],
    version='0.1dev',
    license='AGPLv3',
    long_description=open('README.md').read(),
    install_requires=reqs,
)
