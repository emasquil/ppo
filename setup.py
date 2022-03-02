from setuptools import setup


setup(
    name="ppo",
    version="0.1",
    description="Simple PPO implementation on OpenAI GYM environment.",
    packages=["src"],
    requires=["setuptools", "wheel"],
    install_requires=["dm-env==1.5", "gym==0.22.0", "pygame==2.1.2"],
    extras_require={
        "dev": ["tqdm==4.63.0", "ipykernel==6.9.1", "black==22.1.0"],
    },
)
