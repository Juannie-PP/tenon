from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tenon",
    version="0.1.3",
    package_dir={"": "src"},
    packages=["tenon"],
    url="https://github.com/Juannie-PP/tenon",
    license="MIT",
    author="Juannie-PP",
    author_email="2604868278@qq.com",
    description="A package to solve captcha",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["requests>=2.19.1", "numpy>=1.21.5", "opencv-python==4.5.5.64"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "rotate_captcha",
        "notch_captcha",
        "captcha identify",
        "captcha",
        "verification code",
    ],
    python_requires=">=3.7",
)
