# CS498HS_final_project
#### Data Cleaning:
* **Raw Data**: Go to https://archive.org/details/stackexchange to download `christianity.stackexchange.com.7z`, the `Posts.xml` in it is our data. One can find out the format of `Posts.xml` from `readme.txt` from the same website.
* **Word Embedding Source**: Go to https://github.com/stanfordnlp/GloVe to download `glove.42B.300d.zip`, which contains word-to-vector table.
* **Cleaning**: Run `CS498HS_final_project/data_cleaning.py`. Change the path of `glove.42B.300d.txt` and `Posts.xml` to where you put the file, or put them in `CS498HS_final_project` directory.
* If one don't want to go through the cleaning part, `word_dict_pkl` (word embedding dictionary for words shown in raw data) can be found at [here](https://drive.google.com/file/d/19jQ0s-i897KdvwY7r_wbXGDMZ_bn9csV/view?usp=sharing), load the file to `CS498HS_final_project/resource` folder since `generate_sentence_embedding_file.py` call the file at that place