# Roadmap

**New Features**
- [ ] define on graph transformation a level of granularity for structure extraction
- [ ] estimate a graph size during the inference-duckpunching-phase of the transformation
- [ ] mapping between network models and graphs (partially there)
- [ ] scalable graph themes for transformation into function space

**General**
- [ ] re-work pruning to enable different strategies with an outside-model object-oriented software design
- [ ] document how to extract a single mask
- [ ] document how to initialize a deep directed acyclic network
- [ ] document how to train models with data, e.g. even with pytorch ignite
- [ ] document the graph transformation via duckpunching
- [ ] describe idea of graph themes
- [ ] describe architecture of deepstruct with flowcharts / visualizations
- [ ] describe idea of mapping between network model and graphs (we use networkx)

- [x] sparse recurrent network models
- [x] organize and explain when to use which sparse model in application




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
Execute job *tests*: ``gitlab-runner exec docker test-python3.7``

## Running github action locally
Install *https://github.com/nektos/act*.
Run ``act``

## Running pre-commit checks locally
- Execute pre-commit manually: ``poetry run pre-commit run --all-files``
- Update pre-commit: ``poetry run pre-commit autoupdate``
- Add pre-commit to your local git: ``poetry run pre-commit install``
