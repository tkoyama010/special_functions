from setuptools import setup, find_packages

setup(name='special_functions',
      version='0.0.1',
      description='special functions',
      author='Tetsuo Koyama',
      author_email='tkoyama010@gmail.com',
      url='https://github.com/tkoyama010/special_functions.git',
      packages=find_packages(),
      entry_points="""
      [console_scripts]
      greet = special_functions.special_functions:main
      """,)
