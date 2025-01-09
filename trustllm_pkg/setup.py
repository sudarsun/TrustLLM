from setuptools import setup, find_packages

setup(
    name='trustllm',
    version='0.3.1',
    description='TrustLLM',  
    author='Yue Huang & Siyuan Wu & Haoran Wang & Sudarsun Santhiappan',
    author_email='trustllm.benchmark@gmail.com',
    url='https://github.com/HowieHwong/TrustLLM',  
    packages=find_packages(), 
    include_package_data=True, 
    install_requires=[
        'transformers',
        'huggingface_hub',
        'peft',
        'numpy>=1.18.1',
        'scipy',
        'pandas>=1.0.3',
        'scikit-learn',
        'openai>=1.0.0',
        'tqdm',
        'tenacity',
        'datasets',
        'fschat[model_worker]',
        'python-dotenv',
        'urllib3',
        'anthropic',
        'google.generativeai',
        'google-api-python-client',
        'google.ai.generativelanguage',
        'replicate',
        'zhipuai>=2.0.1',
        'ollama'
],
    classifiers=[
    ],
)
