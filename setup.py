from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str) -> List[str]:
    req = []

    with open(file_path, 'r') as f:
        for line in f:
            req.append(line.strip())
        
        if '-e .' in req:
            req.remove('-e .')
    
    return req

setup(
    name = 'ml-project2',
    version='0.0.1',
    author='Bhaskar Reddy',
    author_email='vanteddubhaskarreddy@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
