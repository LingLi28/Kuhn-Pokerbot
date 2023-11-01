## Setup

Use pip install -r requirements.txt to install dvc

## Useful commands

- use **dvc add** to tell DVC to cache and track them. Example: _dvc add models/model.h5_
- use **dvc push** to upload tracked files. Example: _dvc push models/model.h5.dvc_
- download tracked files or directories from remote storage using **dvc pull**. Example: _dvc pull model.h5.dvc_
- use **dvc checkout** to see data corresponding to the dvc files of that certain commit.

For more info see [DVC](https://dvc.org/ "Named link title")
