from setuptools import setup
import toml

with open("pyproject.toml", "r") as f:
    requirements = toml.loads(f.read())

prod = requirements["install_requires"]
dev = requirements["dev-dependencies"]
examples = requirements["example-dependencies"]
print({"dev": [x + " " + dev[x] for x in dev]})

setup(
    name="DAClib",
    install_requires=[x + "==" + prod[x] for x in prod],
    extras_require={
        "dev": [x + "==" + dev[x] for x in dev],
        "example": [x + "==" + examples[x] for x in examples],
    },
)
