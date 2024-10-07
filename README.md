<div align="center">
<h1>One Piece Image Recognition</h1>
<img src="https://github.com/user-attachments/assets/8be3e0ef-9331-47a9-ad8e-4d7730acbed6"
  height=200
  width=200>
<h3>A machine learning model that can identify images of One Piece characters</h3>
</div>
<hr>

<h4>Requirements: </h4>
<ul>
  <li>Python 3.10</li>
  <li>MatPlotLib 3.9.1</li>
  <li>NumPy 2.0.0</li>
  <li>Pillow 10.4.0</li>
  <li>TensorFlow 2.17.0</li>
  <li>
    <a href="https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier/data?select=Data">One Piece Character Image Set</a>
  </li>
</ul>

**Troubleshooting**:
<ul>
  <li>
    Installing TensorFlow can be finicky, check out the official
    <a href="https://www.tensorflow.org/install/pip">TensorFlow pip install guide if you are having issues</a>.
  </li>
</ul>

**Setup and Usage**:
1. Clone and enter the repo
```sh
git clone git@github.com:greshbasic/OP-Image-Recognition.git
cd OP-Image-Recognition
```
2. Create a venv and install requirements
```sh
python3 -m venv <name>
source <name>/bin/activate
pip install -r requirements.txt
```
3. Run the main.py file
```sh
python3 main.py
```

**What to Expect**:
<ul>
  <li>
    A lot of information is logged to the console, mostly warnings. You can ignore those 
    as they aren't going to negatively impact the model. However, if you want to build upon
    my work it may eventually become something you need to look into.
  </li>
  <li>
    On my mid-range PC build it takes about 2 minutes for the model to fully train.
  </li>
  <li>
    The model averages between 70-75% validation accuracy and is slightly overfit.
  </li>

```sh
Epoch 1/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 26s 58ms/step - accuracy: 0.0551 - loss: 2.9549 - val_accuracy: 0.0739 - val_loss: 2.8883 - learning_rate: 0.0010
Epoch 2/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 8s 26ms/step - accuracy: 0.0798 - loss: 2.8492 - val_accuracy: 0.1483 - val_loss: 2.6521 - learning_rate: 0.0010
Epoch 3/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.1881 - loss: 2.5654 - val_accuracy: 0.2438 - val_loss: 2.3851 - learning_rate: 0.0010
Epoch 4/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.2864 - loss: 2.2582 - val_accuracy: 0.3511 - val_loss: 2.1002 - learning_rate: 0.0010
Epoch 5/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.3707 - loss: 1.9949 - val_accuracy: 0.4432 - val_loss: 1.8730 - learning_rate: 0.0010
Epoch 6/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.4450 - loss: 1.7575 - val_accuracy: 0.4818 - val_loss: 1.7609 - learning_rate: 0.0010
Epoch 7/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.5161 - loss: 1.5475 - val_accuracy: 0.5028 - val_loss: 1.6314 - learning_rate: 0.0010
Epoch 8/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.5589 - loss: 1.3771 - val_accuracy: 0.5665 - val_loss: 1.4851 - learning_rate: 0.0010
Epoch 9/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.6136 - loss: 1.2008 - val_accuracy: 0.5761 - val_loss: 1.4158 - learning_rate: 0.0010
Epoch 10/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.6428 - loss: 1.1050 - val_accuracy: 0.6074 - val_loss: 1.3171 - learning_rate: 0.0010
Epoch 11/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.6845 - loss: 0.9817 - val_accuracy: 0.6313 - val_loss: 1.2414 - learning_rate: 0.0010
Epoch 12/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.7127 - loss: 0.8847 - val_accuracy: 0.6097 - val_loss: 1.2567 - learning_rate: 0.0010
Epoch 13/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.7490 - loss: 0.7974 - val_accuracy: 0.6585 - val_loss: 1.1779 - learning_rate: 0.0010
Epoch 14/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.7575 - loss: 0.7488 - val_accuracy: 0.6562 - val_loss: 1.1683 - learning_rate: 0.0010
Epoch 15/15
312/312 ━━━━━━━━━━━━━━━━━━━━ 7s 22ms/step - accuracy: 0.7769 - loss: 0.6842 - val_accuracy: 0.6784 - val_loss: 1.1223 - learning_rate: 0.0010
```
  <li>
    Users will be shown 9 random images and the model's classification of each, alongside the training history
  </li>
</ul>
<img src="https://github.com/user-attachments/assets/d6cf201d-8649-4dec-9dad-185a3b15124d"
  width=480
  height=480>
<img src="https://github.com/user-attachments/assets/79672fd2-9f98-4c9c-b737-929790cdbdcf"
  width=480
  height=480>

<div align="center">
<img src="https://github.com/user-attachments/assets/97957567-6003-4c15-843c-d0cee01dae21"
  width=480
  height=480>
<hr>
<strong>Special Thanks</strong>: I want to give a huge thanks to Ibrahim Serouis on Kaggle for creating the One Piece character image data set!
</div>

