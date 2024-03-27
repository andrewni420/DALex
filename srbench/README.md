
# SRBench benchmark suite
This section tests DALex versus epsilon lexicase on black-box regression problems from the SRBench suite. Code is adapted from [gplearn](https://github.com/trevorstephens/gplearn/tree/main) by Trevor Stephens and [Contemporary Symbolic Regression Methods and their Relative Performance](https://arxiv.org/abs/2107.14351) by La Cava et al. (2021).

# Usage

This benchmark suite relies on using PMLB datasets from [PMLB v1.0: an open source dataset collection for benchmarking machine learning methods](https://arxiv.org/abs/2012.00058) by Romano et al. (2020), which are not distributed with this repository. The datasets should be downloaded and unzipped before running experiments.

To run symbolic regression, do

`python analyze.py "path/to/dataset" -ml [method] --local -skip_tuning`

[method] should be the name of a folder in the `/srbench/methods` folder sans the ".py" extension, and can be `gplearn` for gplearn with tournament selection (the default), `epsilon_lexicase` for gplearn with epsilon lexicase selection, or `dalex` for gplearn with DALex. 
