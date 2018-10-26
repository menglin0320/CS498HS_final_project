import numpy as np

def padding_samples(passage_embedding, max_n_embed):
    padding_needed = max_n_embed - len(passage_embedding)
    if padding_needed < 0:
        raise(ValueError('max embedding length evaluate fault'))
    for i in range(0, padding_needed):
        passage_embedding.append([0]*300)
    return passage_embedding

def cvt2onehot(values):
    n_values = np.max(values) + 1
    ret = np.eye(n_values)[values]
    return ret

def get_batches(data_, batch_size):
    ret = []
    i = 0
    while(batch_size*i < len(data_)):
        cur_batch = {}
        cur_batch_start = len(data_) - i*batch_size
        cur_batch_size = max(batch_size, len(data_) - i*batch_size)
        max_n_embed = len(data_['attributes'][cur_batch_start + cur_batch_size - 1])
        cur_batch['data'] = []
        cur_batch['label'] = []
        cur_batch['mask'] = []
        for data_ind in range(cur_batch_start, cur_batch_start + cur_batch_size):
            cur_batch['mask'].append(len(data_['attributes'][data_ind])-1)
            cur_batch['data'].append(padding_samples(data_['attributes'][data_ind], max_n_embed))
            cur_batch['label'].append(data_['label'][data_ind])
        cur_batch['mask'] = cvt2onehot(np.asarray(cur_batch['mask']))
        cur_batch['label'] = np.asarray(cur_batch['label'])
        cur_batch['data'] = np.asarray(cur_batch['data'])
        ret.append(cur_batch)
        i+=1
    return ret