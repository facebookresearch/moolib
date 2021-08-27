# moolib

## Installing

```
git clone --recursive https://github.com/fairinternal/moolib
cd moolib
pip install -e .
```
This will also install pytorch if necessary.

Run `python test/test.py` to see if it works.


## Development mode

For development without CUDA and compilation in debug mode:

```
USE_CUDA=0 python setup.py build --debug install
```
