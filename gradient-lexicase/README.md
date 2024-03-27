## Basic usage

To train and evaluate baseline architectures, do

`python3 base.py`

To train and evaluate gradient lexicase selection, do

`python3 lexicase.py`

To train and evaluate gradient DALex, do 

`python3 dalex.py --selector [selector] --std [particularity_pressure]`

\[selector\] can be "dalex" or "dalex accuracy" to use accuracies or cross entropy losses as the error function. \[particularity_pressure\] controls the relaxed-ness of selection, and should be set to a high number, such as 200.

The architectures are indexed as follows (from 0 to 1): VGG, ResNet18

Please contact the authors for further questions.
