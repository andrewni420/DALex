# Learning Classifier Systems

This code compares lexicase selection versus DALex in Learning Classifier Systems. The base code is derived from Ryan Urbanowicz's implementation of [eLCS] (https://github.com/ryanurbs/eLCS) and [Lexicase Selection in Learning Classifier Systems](https://dl.acm.org/doi/abs/10.1145/3321707.3321828) by Aenugu & Spector (2019)

## Running the code
<ul>
<li> Few demo datasets taken from the [Penn-ML-Benchmarks](https://github.com/EpistasisLab/penn-ml-benchmarks) used for studying the parent selection techniques are in the `Demo_Datasets` folder </li>
<li> The LCS hyperparameter settings including the parent selection techniques can be set in the Configuration file (`eLCS_Configuration_File.py`)</li>
  <li> Available selection techniques include "dalex", "batch-lexicase", "lexicase", "tournament", and "roulette"</li>
</ul> Run the algorithm using `python eLCS_Run.py`
