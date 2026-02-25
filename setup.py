from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Image-based product attribute extraction pipeline using an ensemble of MiniCPM-2.6, Qwen2-VL-7B (LLaMA-Factory fine-tuned), combining zero-shot, few-shot, and fine-tuned strategies for robust entity extraction.',
    author='Mukul Bhele',
    license='',
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.37.0',
        'pandas>=2.0.0',
        'Pillow>=10.0.0',
        'requests>=2.31.0',
        'tqdm>=4.65.0',
    ],
    entry_points={
        'console_scripts': [
            'entity-extract=main:main',
        ],
    },
)
