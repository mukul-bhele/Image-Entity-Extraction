from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Developed an image-based product attribute extraction pipeline using an ensemble of MiniCPM-2.6, Qwen2-VL-7B (LLaMA-Factory fine-tuned), and InternVL2-8B, combining zero-shot, few-shot, and fine-tuned strategies for robust entity extraction.',
    author='Mukul Bhele',
    license='',
)
