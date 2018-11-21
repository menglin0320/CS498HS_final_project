import untangle
import pickle
from bs4 import BeautifulSoup
import numpy
import nltk
from nltk import tokenize, word_tokenize
import os

#get embedding table
word_set = dict()
with open('glove.42B.300d.txt', encoding='utf8') as glove_file:
    sample = glove_file.readline()
    i=0
    while sample!='':
        word_set[sample.split(' ', 1)[0]]=sample.split(' ', 1)[1]
        i+=1
        if i%10000==0:
            print(i)
        sample = glove_file.readline()
    #print(word_set)
    glove_file.close()

# abbriviation_replace to avoid wrong sentence slicing (due to misinterpreted meaning of '.')
def abbriviation_replace(sentence):
    abbreviations = {'dr.': 'doctor', 'mr.': 'mister', 'bro.': 'brother', 'mrs.': 'mistress', 'ms.': 'miss', 'jr.': 'junior', 'sr.': 'senior', 'i.e.': 'for example', 'e.g.': 'for example', 'vs.': 'versus'}
    for key in abbreviations:
        sentence = sentence.replace(key, abbreviations[key])
    return sentence


# change Posts.xml into List of data records (saved in data_pkl) and also output a word_pkl file.
# word_pkl contains a Dictionary having word(string object) as key and vector(List of floats) as value. It is also a subset of glove.42B.300d.txt
# Data record format: [Body(string), accepted tag(bool)]

obj = untangle.parse('Posts.xml')
questionDict = {} # questionDict[id] = false ,if this question has not choose accepted answer
                  #                    N     ,if this question choose post with id N as the accepted answer
word_dict = {}
answers = []
#for data statistic
acc_count = 0
not_acc_count = 0
not_rate_count = 0
weird_ans_count = 0
question_count = 0
answer_count = 0
non_matched_num = 0
non_matched = {}

for i in range(len(obj.posts.row)):
    #progress bar
    if i%1000==0:
        print("Object {}".format(i))
    
    #question type post
    if obj.posts.row[i]['PostTypeId'] == '1':
        question_count += 1
        questionDict[obj.posts.row[i]['Id']] = bool(obj.posts.row[i]['AcceptedAnswerId'])
        if bool(obj.posts.row[i]['AcceptedAnswerId']):
            questionDict[obj.posts.row[i]['Id']] = obj.posts.row[i]['AcceptedAnswerId']
    #answer type post
    else:
        answer_count += 1
        #has no attribute of 'ParentId' or ParentId not found
        if obj.posts.row[i]['ParentId'] not in questionDict:
            weird_ans_count += 1
        #parent question has choose accepted answer
        elif questionDict[obj.posts.row[i]['ParentId']]:
            #data cleaning
            raw = BeautifulSoup(obj.posts.row[i]['Body'], "lxml").get_text()
            raw = raw.lower()
            raw = abbriviation_replace(raw)
            #remove words that cannot found in word_embbeding glove file
            sentences = tokenize.sent_tokenize(raw)
            raw = []
            for s in sentences:
                raw_words = word_tokenize(s)
                ok_words = []
                for w in raw_words:
                    if w in word_set and w not in word_dict:
                        word_dict[w] = word_set[w]
                    if w in word_set:
                        ok_words += [w]
                    # statistic
                    elif w not in word_set:
                        non_matched_num += 1
                        if w in non_matched:
                            non_matched[w] += 1
                        else:
                            non_matched[w] = 1
                raw += [" ".join(ok_words)]
            raw = " ".join(raw)
            acc = (questionDict[obj.posts.row[i]['ParentId']] == obj.posts.row[i]['Id'])
            # statistic
            acc_count += acc
            not_acc_count += 1 - acc
            #define data record format here
            answers += [[raw, acc]]
        #parent question has not choose accepted answer
        else:
            not_rate_count += 1

# statistic
rated_question=0
for key in questionDict:
    if questionDict[key]:
        rated_question += 1

# assertion part
"""
if(question_count!=len(questionDict)):
    print("Repeat question,{}!={}".format(question_count,len(questionDict)))
if question_count + answer_count != len(obj.posts.row):
    print("entity mismacth,{}+{}={}!={}".format(question_count, answer_count, question_count + answer_count, len(obj.posts.row)))
if acc_count + not_acc_count + not_rate_count + weird_ans_count != answer_count:
    print("entity mismacth,{}+{}+{}+{}={}!={}".format(acc_count, not_acc_count, not_rate_count, weird_ans_count,
        acc_count + not_acc_count + not_rate_count + weird_ans_count, answer_count))
"""

# statistic
print("questions:{},questions choose accepted answer:{}, answers:{},accepted answers:{},not accepted answers:{},not rated answers:{}, no matching question answers:{}".format(
    len(questionDict),rated_question, len(obj.posts.row)-len(questionDict), acc_count, not_acc_count, not_rate_count, weird_ans_count))
print("match failed words number:{}, kinds:{}".format(non_matched_num, len(non_matched)))
print(answers[0])
# output files
current_dir = os.path.dirname(os.path.abspath(__file__))

train_out_path = current_dir + "/data/processed_data/data_pkl"
with open(train_out_path, 'wb') as trainfile:
    pickle.dump(answers, trainfile)

word_dict_out_path = current_dir +"/resource/word_dict_pkl"
with open(word_dict_out_path, 'wb') as word_file:
    pickle.dump(word_dict, word_file)

