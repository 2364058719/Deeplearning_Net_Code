#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
from bert_sup import *


#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab



if __name__ == '__main__':
    #@tab pytorch
    d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                                '225d66f04cae318b841a13d32af3acc165f253ac')
    d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                                'c72329e68a732bef0452e4b96a1c341c8910f81f')
    #@tab all
    devices = d2l.try_all_gpus()
    bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
            num_blks=2, dropout=0.1, max_len=512, devices=devices)
    #@tab pytorch
    # Reduce `batch_size` if there is an out of memory error. In the original BERT
    # model, `max_len` = 512
    batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
    # data_dir = d2l.download_extract('SNLI')
    data_dir = "..\data\snli_1.0"
    train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                    num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                    num_workers=num_workers)
    #@tab pytorch
    net = BERTClassifier(bert)
    #@tab pytorch
    lr, num_epochs = 1e-4, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    net(next(iter(train_iter))[0])
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    d2l.plt.show()