# INSTALLING PESDT

Either using git, or just downloading the .zip file from github (https://github.com/Edge-Physics-Group/PESDT/), copy the files to your home. In your home you should have the following structure:

*/home/<your_user_name>/PESDT/*


## INSTALLING CHERAB, AND THE PESDT ADD-ON MODULE FOR CHERAB (JDC)
It is recommended to use python/3.9, other versions might work (newer), but this one is proven to be working. (June 26th: Newer versions of Python (e.g) 3.13 have been made available, but installing numpy is impossible, because the default compiler python uses on the JDC is too old.) To use python/3.9, do (or add to your .bashrc)

*module unload python/<your current version> *

*module load python/3.9*

To install Cherab, you'll need a functioning version of the "cython module".

*pip install Cython>=3.1.2* 

To install Cherab, first install the core package with pip:

*pip install cherab* 


Finally, the PESDT add-on module for Cherab, which adds support for AMJUEL data and molecules, is installed similarly via pip,

*pip install ./PESDT/PESDT_addon-master*

This completes the installation process. Note that to use AMJUEL data, you need to supply the AMJUEL.tex data file yourself. Currently, PESDT assumes that AMJUEL.tex is located in the user home folder, but you may point to the correct directory by setting the environment variable **AMJUEL_PATH** to the AMJUEL directory.

# RUNNING PESDT

In "PESDT/inputs/" you have input .json files. **DO NOT** edit the adf11 or adf15 files, unless you explicitly know what you're doing. In the "PESDT/inputs/" there is a commented example file, which explains each of the lines. By default the JSON format does not support comments, that is, you need to make an input deck of your own *based on* the example input decks. In the input file, you should define your edge-code, the path to the simulation results, the spectroscopic lines you wish to analyze, and the instruments you want to synthesize. Also add your save directory. (**NOTE (JDC/EDGE2D)**: eproc wants that the tranfile is named just "tran", nothing else will do.)

There are two modes for running PESDT, the basic mode, which uses cone-integrals to calculate the emissions, and ray-tracing mode, which uses Cherab to determine the emission *including* reflections. To use Cherab via PESDT, simply set the flag

*"run_cherab": true*

in the PESDT input deck. Note that ray-tracing can be resource intensive, and it is adviced that the PESDT run is submitted to the batch system.

To run PESDT, open a new terminal, and run the PESDT_run.py script with the input deck JSON file as a command line argument,

*python3 PESDT/PESDT_run.py PESDT/inputs/<your_inputfile_name>.json*

(i.e. after specifying the run script, add as an argument your input .json file)

PESDT should now run, and you should find, after a while, the output pickle and .json files under the *savedir/<case>*.


# PLOTTING

The plotting scripts are a mess, and need to be cleaned up. Use your own to read and plot the result JSON files.

# PESDT environment variables

**PESDTCacheDir**: points to the folder where PESDT should cache things like the mesh, or ADAS data

**AMJUEL_PATH**: points to the directory where AMJUEL.tex is located


