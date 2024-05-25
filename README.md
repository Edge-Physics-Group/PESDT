INSTALLING PESDT

Either using git, or just downloading the .zip file from github (For now from: https://github.com/vrikala/PESDT, in the future https://github.com/lomanowski/PESDT), copy the files to your home. In your home you should have the following structure:

/home/<your_user_name>/PESDT/

in your .bashrc (if on JET-JDC), you should have the following lines:

export PYTHONPATH=$PYTHONPATH:"/home/adas/python/"
export PYTHONPATH=$PYTHONPATH:$HOME/PESDT
export PYTHONPATH=$PYTHONPATH:"/u/sim/jintrac/v280818/libs/eproc/python/eproc"
export PYTHONPATH=$PYTHONPATH:"/u/sim/jintrac/v280818/libs/eproc/python"
export CHERAB_CADMESH='/common/cadmesh/'

Alternatively, source the example bashrc-file included in the PESDT package
These lines loads the adas and adaslib python routines, PESDT (the program you're about to use), and eproc (used for reading the EDGE2D-EIRENE tranfiles) as importable python libraries. The cherab cadmesh is the path to the JET wall cadmesh files, used by cherab for raytracing

You also need a working installation of cherab. 

'''
On 24/07/2023 PESDT switched to the 1.40 version of cherab. Older versions won't work with this edition of PESDT.
'''

To install Cherab, you'll need a functioning version of the "cython module"

pip install Cython (Note: pre-release version 3.0.5a was used, but the new default 3.0.2 should work)

To install Cherab, first install the core package with pip:

pip install cherab

Then, you'll need the cherab/edge2d and cherab/jet modules. You can find the files on github. Download the folders to your home, and install via pip: (example)

pip install ./cherab-edge2d/ --user

(here, cherab-edge2d should be the folder containing the setup.py file)

Repeat for cherab-jet module


USING PESDT

PESDT works by generating a background pickle file, and json files of the synthetic diagnostics.
In "PESDT/inputs/" you have input .json files. DO NOT edit the adf11 or adf15 files, unless you explicitly know what you're doing. In the "PESDT/inputs/" there is an commented example file, which explains each of the lines. For some unknown reason, currently any comments in the actual input break the code, so you should only use it as a reference. In the input file, you should define your EDGE2D-EIRENE (or SOLPS, currently untested) tranfile location, the spectoscopic lines, and the instruments you want to synthesize. (Also add your save directory)

NOTE: eproc wants that the tranfile is named just "tran", nothing else will do.

In the terminal, in your home folder, type and enter:

ipython

and in the interactive python environment type and enter:

run PESDT/run_PESDT.py PESDT/inputs/<your_inputfile_name>.json

(i.e. after specifying the run script, add as an argument your input .json file)

PESDT should now run, and you should find, after a while, the ouput pickle and json files under the savedir/<case>

For plotting, edit plot_PESDT.py file to point to your output files, and add the lines/instruments you want to plot.

REFLECTIONS

To get the impact of reflections, we need to run the raytracing code cherab. Cherab is linked to PESDT via cherab bridge, which loads in the PESDT generated background plasma, and wraps cherab code. 

An example inputfile for cherab bridge is located in PESDT/cherab_bridge. Edit it to point to your PESDT result files.

RUNNING CHERAB_BRIDGE

Ray tracing is quite slow, it is recommended to use slurm (submit job) to run cherab bridge.

TODO: WRITE SECTION


