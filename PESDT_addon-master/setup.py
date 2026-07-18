from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

compilation_includes = [".", numpy.get_include()]

setup_path = path.dirname(path.abspath(__file__))

# build extension list
extensions = []
for root, dirs, files in os.walk(setup_path):
    for file in files:
        if path.splitext(file)[1] == ".pyx":
            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")
            extensions.append(
                                Extension(
                                    module,
                                    [pyx_file],
                                    include_dirs=compilation_includes,
                                    libraries=["m", "mvec"],
                                    extra_compile_args=["-O3", "-ffast-math", "-march=native", "-std=c99"],
                                    extra_link_args=[],
                                )
)


if profile:
    directives = {"profile": True}
else:
    directives = {}


setup(
    name="cherab-PESDT_addon",
    version="0.1.0",
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    packages=find_packages(),
    install_requires=["numpy>=2.0", "scipy", "cherab==1.5.0.1", "raysect>=0.9.1", "Cython>=3.0"],
    include_package_data=True,
    package_data={
        "": ["*.pxd", "*.pyx"]
    },
    ext_modules=cythonize(extensions, force=force, compiler_directives=directives, language_level=3)
)
