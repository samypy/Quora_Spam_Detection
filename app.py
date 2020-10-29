import torch
from torch.nn import BCEWithLogitsLoss
from transformers import XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, jsonify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)


# config = XLNetConfig()

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=2):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, \
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids, \
                                       attention_mask=attention_mask, \
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), \
                            labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state


model = XLNetForMultiLabelSequenceClassification()


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

model = torch.load('model\quora.pt',map_location=map_location)
model.eval()
def predictions(text):
  label_cols = ['target']
  probs = np.array([]).reshape(0, 1)
  input_ids = tokenize_inputs(text, tokenizer, num_embeddings=250)
  attention_masks = create_attn_masks(input_ids)
  #model.to(device)
  #model.eval()
  X = input_ids
  masks = attention_masks
  X = torch.tensor(X).to(torch.int64)
  #X = torch.tensor(X)
  masks = torch.tensor(masks, dtype=torch.long)
  X = X.to(device)
  masks = masks.to(device)
  with torch.no_grad():
      logits = model(input_ids=X, attention_mask=masks)
      logits = logits.sigmoid().detach().cpu().numpy()
      probs = np.vstack([probs, logits])
      if probs > 0.5:
          pred =  "Oh No! It's a spam"
      else:
          pred = "Great! It's not a spam"

  return pred

app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    text = request.form['question']
    if text == "":
        return render_template('index.html', prediction_text='Please enter your question')
    else:
        mytext = [text]
        print(mytext)
        prediction = predictions(mytext)
    return render_template('index.html', prediction_text= prediction)

if __name__ == "__main__":
    app.run(debug=True)