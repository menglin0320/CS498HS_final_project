# CS498HS_final_project
# Requirements
supported python version 3.5

Environment Ubuntu 16.04

To run the repo successfully on Mac or Windows, you have to modify the system calls and shell scripts to meet your environment.

Open command run:
```shell
sudo sh ./installation.sh
```

# Usage
For cross validation, run
```shell
python3 cross_validation.py 10
```
Notice that this cross validation takes days on a laptop. And the script can be resource demanding, it's better to run it on a lab machine.

For the following two steps, you have to run:
```shell
python3  generate_sentence_embedding_file.py
```
to get the splited train validation data in sentence embedding form first.
(if you already run cross_validation, this step is not required since train valid split are preformed ten times on cross validation)

if you quit from cross validation, clean save folder before running the following python files.

to train the model, run
```shell
python3  train.py
```

to test the model, run
```shell
python3  evaluate.py data/pkl_test_data
```
a output.txt file will be generated after excuting evaluate.py.

# Directly run evaluate without training:
You can open the link 
```shell
https://drive.google.com/open?id=1K2i_6PiZ4XztVDWU_svQ3n1JVvoB5qU6
```
Download the zip file put it on root of the project and then run

```shell
sh  directly_run.sh
```
then run
```shell
python3  evaluate.py data/pkl_test_data
```

# How do we get the data
#### Data Cleaning:
* **Raw Data**: Go to https://archive.org/details/stackexchange to download `christianity.stackexchange.com.7z`, the `Posts.xml` in it is our data. One can find out the format of `Posts.xml` from `readme.txt` from the same website.
* **Word Embedding Source**: Go to https://github.com/stanfordnlp/GloVe to download `glove.42B.300d.zip`, which contains word-to-vector table.
* **Cleaning**: Run `CS498HS_final_project/data_cleaning.py`. Change the path of `glove.42B.300d.txt` and `Posts.xml` to where you put the file, or put them in `CS498HS_final_project` directory.
* If one don't want to go through the cleaning part, `word_dict_pkl` (word embedding dictionary for words shown in raw data) can be found at [here](https://drive.google.com/file/d/19jQ0s-i897KdvwY7r_wbXGDMZ_bn9csV/view?usp=sharing), load the file to `CS498HS_final_project/resource` folder since `generate_sentence_embedding_file.py` call the file at that place