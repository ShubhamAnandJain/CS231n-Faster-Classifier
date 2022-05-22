conda env create -f environment.yml
conda activate fasterClassifier
python -m ipykernel install --user --name fasterClassifier
cd approx/src/pytorch/cpp
python setup.py install
cd ../../../../
