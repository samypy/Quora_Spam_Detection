# Spam filter for Quora questions
Goal : Build a model for identifying if a question on Quora is spam

Please use following link to download dataset.
https://www.dropbox.com/sh/kpf9z73woodfssv/AAAw1_JIzpuVvwteJCma0xMla?dl=0

Please use following link to download pretrained model and place it under model folder.
https://drive.google.com/file/d/1--2NXb5ZXzGaW0EZVWdpDoA2K7wWSD-I/view?usp=sharing

#Model Comparision:

1) Base Model(RNN/LSTM):
    val_loss: 0.3839 - val_accuracy: 0.8834
    roc_auc_score - 0.9326992592997551
    
2) RNN/LSTM with pre trained Glove Embedding:
    val_loss: 0.2691 - val_accuracy: 0.8961
    roc_auc_score - 0.6893690089633167
    
3) XLnet Model:
    Val_loss: 0.014194569230107968
    roc_auc_score - 0.9820409516568424

Xlnet Mode outperformed other two models and hence we have considered XLnet to solve our problem statement.

