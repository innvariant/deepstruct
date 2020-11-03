# Roadmap

**New Features**
- [ ] sparse recurrent network models
- [ ] mapping between network models and graphs (partially there)
- [ ] scalable graph themes for transformation into function space

**General**
- [ ] describe idea of graph themes
- [ ] describe architecture of deepstruct with flowcharts / visualizations
- [ ] describe idea of mapping between network model and graphs (we use networkx)
- [x] organize and explain when to use which sparse model in application


# Architecture
Flat & simple :)

# Practices & Conventions

## Publishing
```bash
poetry build
twine upload dist/*
```
- Create wheel files in *dist/*: ``poetry build``
- Install wheel in current environment with pip: ``pip install path/to/deepstruct/dist/deepstruct-0.1.0-py3-none-any.whl``

## Running CI image locally
Install latest *gitlab-runner* (version 12.3 or up):
```bash
# For Debian/Ubuntu/Mint
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash

# For RHEL/CentOS/Fedora
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash

apt-get update
apt-get install gitlab-runner

$ gitlab-runner -v
Version:      12.3.0
```
Execute job *tests*: ``gitlab-runner exec docker test-python3.6``

## Running github action locally
Install *https://github.com/nektos/act*.
Run ``act``

## Running pre-commit checks locally
- Execute pre-commit manually: ``poetry run pre-commit run --all-files``
- Add pre-commit to your local git: ``poetry run pre-commit install``
