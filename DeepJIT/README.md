# DeepJIT: An End-To-End Deep LearningFramework for Just-In-Time Defect Prediction [[pdf](https://posl.ait.kyushu-u.ac.jp/~kamei/publications/Thong_MSR2019.pdf)]

## Contact
Questions and discussion are welcome: vdthoang.2016@smu.edu.sg

## Implementation Environment

Please install the neccessary libraries before running our tool:

- python==3.6.9
- torch==1.2.0
- tqdm==4.46.1
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data & Pretrained models:

Please following the link below to download the data and pretrained models of our paper. 

- https://drive.google.com/drive/folders/1sMl2-LbVi3__56kMeS8YSUW7qaKrL2sQ?usp=sharing
- https://zenodo.org/record/3965246#.XyEDVnUzY5k

After downloading, simply copy the data and model folders to DeepJIT folder. 

## Hyperparameters:
We have a number of different parameters

* --embedding_dim: Dimension of embedding vectors.
* --filter_sizes: Sizes of filters used by the convolutional neural network. 
* --num_filters: Number of filters. 
* --hidden_layers: Number of hidden layers. 
* --dropout_keep_prob: Dropout for training. 
* --l2_reg_lambda: Regularization rate. 
* --learning_rate: Learning rate. 
* --batch_size: Batch size. 
* --num_epochs: Number of epochs. 

## Running and evalutation
      
- To train the model for just-in-time defect prediction, please follow this command: 

      $ python main.py -train -train_data [path of our data] -dictionary_data [path of our dictionary data]
      
- To evaluate the model for just-in-time defect prediction, please follow this command:
      
       $ python main.py -predict -pred_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model]

## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg
