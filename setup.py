#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# To install: pip install .
#
# For debug builds: python setup.py build --debug install
#
# The environment variable USE_CUDA can be set to "OFF" (or 0).
#

import os
import pathlib
import subprocess
import sys

import setuptools
from setuptools.command import build_ext
from distutils import spawn


class CMakeBuild(build_ext.build_ext):
    def run(self):  # Necessary for pip install -e.
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        source_path = pathlib.Path(__file__).parent.resolve()
        output_path = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        os.makedirs(self.build_temp, exist_ok=True)

        build_type = "Debug" if self.debug else "RelWithDebInfo"

        generator = "Ninja" if spawn.find_executable("ninja") else "Unix Makefiles"

        cmake_cmd = [
            "cmake",
            str(source_path),
            "-G%s" % generator,
            "-DCMAKE_BUILD_TYPE=%s" % build_type,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%s" % output_path,
        ]

        use_cuda = os.environ.get("USE_CUDA", True)
        if use_cuda == "OFF":
            use_cuda = False
        if not int(use_cuda):
            cmake_cmd.append("-DUSE_CUDA=OFF")

        build_cmd = ["cmake", "--build", ".", "--parallel"]

        # pip install (but not python setup.py install) runs with a modified PYTHONPATH.
        # This can prevent cmake from finding the torch libraries.
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"]
        try:
            subprocess.check_call(cmake_cmd, cwd=self.build_temp, env=env)
            subprocess.check_call(build_cmd, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError:
            # Don't obscure the error with a setuptools backtrace.
            sys.exit(1)


def main():
    with open("README.md") as f:
        long_description = f.read()

    setuptools.setup(
        name="moolib",
        version="0.0.9",
        description=("A library for distributed ML training with PyTorch"),
        long_description=long_description,
        author="tscmoo & the moolib dev team",
        url="https://github.com/facebookresearch/moolib",
        classifiers=[
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS :: MacOS X",
            "Environment :: GPU :: NVIDIA CUDA",
        ],
        packages=["moolib", "moolib.examples.common", "moolib.examples.vtrace"],
        package_dir={"": "py", "moolib.examples": "examples"},
        ext_modules=[setuptools.Extension("moolib._C", sources=[])],
        install_requires=["torch>=1.6.0"],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
