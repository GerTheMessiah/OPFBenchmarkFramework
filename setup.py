from setuptools import setup


setup(
    install_requires=[
        'numpy==1.23.5',
        'scipy',
        'numba==0.56.4',
        'pandas',
        'matplotlib',
        'pandapower==2.13.1',
        'simbench==1.4.0',
        'google-auth==2.22.0',
        'gymnasium',
        'pytest',
        'pettingzoo',
        'ray',
        'dm_tree',
        'scikit-image',
        'lz4',
        'tqdm',
        'gputil',
        # 'tensorboardX',
        'tensorboard',
        'tensorflow_probability',
        'fsspec',
        'pyarrow',
        'tabulate'
    ],
)
