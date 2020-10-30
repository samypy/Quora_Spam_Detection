# Spam filter for Quora questions
Goal : Build a model for identifying if a question on Quora is spam

Please use following link to download dataset.
https://www.dropbox.com/sh/kpf9z73woodfssv/AAAw1_JIzpuVvwteJCma0xMla?dl=0

Please use following link to download pretrained model and place it under model folder.


XLnet Model : https://drive.google.com/file/d/1--2NXb5ZXzGaW0EZVWdpDoA2K7wWSD-I/view?usp=sharing

#Model Comparision:

1) Base Model(RNN/LSTM):
    val_loss: 0.2706  - val_accuracy: 0.8945
    roc_auc_score - 0.9326992592997551
    
2) Bidirectional_LSTM with pre trained Glove Embedding:
    val_loss: 0.2516 - val_accuracy: 0.9022
    roc_auc_score - 0.7283488761858034
    
3) XLnet Model:
    val_loss: 0.014194569230107968
    test_accuracy: 0.9437157117518459
    roc_auc_score - 0.9820409516568424

# Note: You can find the script which was used to train these models under Model_training folder.

Xlnet Model outperformed other two models and hence I have considered XLnet to solve our problem statement.

