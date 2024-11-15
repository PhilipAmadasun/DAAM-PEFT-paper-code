This directory has some test scripts on how to implement DAAM to make an SER model.

# Procedure
This initital tutorial was developed with the `microsoft/wavlm-large` which was trained on 4 emotion categories. So the dataset used to train the aforementioned modes
was also used to train the DAAM model. If you are interested in another model that was trained on different datasets and/or has different emotion categories and so on, be mindful of this.
In the future I plan to implement DAAM on and SER model that outputs valence, arousal and dominance scores as those seeem to be the most accurate.

These ae the steps:
* You will need the `adapted_IEMOCAP` and `IEMOCAP_full_release`. You will need to ask for access for the former, the later you can get from kaggle.
* Now run the `preprocess_features.py` script which vectorizes the .wav files i the data set to tensors which contain voice characteristics.
* The  run the `daam_test.py` to train the DAAM model. The model is mase up of  a DAAM block, then 2 convolution layers, then a fully connected layer.
  * This is simply a test so the model is not optimal and gave poor results (0.64 for best check point I believe)
* `daam_model_test.py` can be run to test a checkpoint
