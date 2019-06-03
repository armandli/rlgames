from setuptools import setup, find_packages

setup(
  name='rlgames',
  version='0.0.1',
  description='Reinforcement learning for games',
  author='armandli',
  author_email='armand.li@hotmail.com',
  packages=find_packages(),
  package_data={},
  data_files=[],
  install_requires=[],
  entry_points={},
  scripts=['rlgames/bot_vs_bot.py', 'rlgames/human_vs_bot.py', 'rlgames/bot_vs_bot_ttt.py', 'rlgames/human_vs_bot_ttt.py', 'rlgames/index_processor.py']
)
