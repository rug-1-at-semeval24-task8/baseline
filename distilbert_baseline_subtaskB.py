import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf
import torch

def main():
    file_path="./dataset/SubtaskB/"
    train_df = pd.read_json(path_or_buf=file_path+"subtaskB_train.jsonl", lines=True)
    train_texts = train_df["text"].to_list()
    train_labels = train_df["label"].to_list()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size = 0.2, random_state = 42 )
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation = True, padding = True  )
    val_encodings = tokenizer(val_texts, truncation = True, padding = True )
    print(len(train_texts))
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))
    training_args = TFTrainingArguments(
    output_dir='./results',          
    num_train_epochs=10,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,   
    warmup_steps=100,                
    weight_decay=1e-5,               
    logging_dir='./logs',            
    eval_steps=500                   
    )
    with training_args.strategy.scope():
        trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 6 )

    trainer = TFTrainer(
        model=trainer_model,                 
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,            
    ) 
    trainer.train()
    trainer.evaluate()   
    save_directory="/saved_models/distil_baseline_B"
    trainer_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    test_df = pd.read_json(path_or_buf=file_path+"subtaskB_dev.jsonl", lines=True)
    test_texts = test_df["text"].to_list()
    test_labels = test_df["label"].to_list()
    torch.inference_mode()

    predict_input_pt = tokenizer(test_texts, truncation = True, padding = True, return_tensors = 'pt' )
    ouput_pt = trainer_model(predict_input_pt)
    prediction_value_pt = torch.argmax(ouput_pt[0], dim = 1 ).item()
    print(classification_report(test_labels, prediction_value_pt))
    print(confusion_matrix(test_labels, prediction_value_pt))
    with open('/predictions/distil_bert.txt', 'w') as f:
        f.write(prediction_value_pt)

if __name__ == '__main__':
    main()