# Text-Generation-using-LSTM-RNN
The text data(Peter Pan in this example) is from Project Gutenberg

<h2>How to run</h2>

<h3>Below command will train the model and save the model weights</h3>
python train.py

<h3>Generating the texts</h3>
<ul>
  <li>Provide the name of the saved model with -w keyword followed by the program name </li>
</ul>
e.g. python test.py -w model-weights-01-2.7223.hdf5

<h3>Additional Info</h3>
It took more than 6days with my laptop, in which the cpu is 'Intel(R) Core(TM i5-8250U CPU' <br />
However, less than 2hours with RTX 3070 ti with the following hyper-parameters

num. of epochs: 50 <br />
batch_size: 64 <br />
optimizer: adam <br />
loss: categorical crossentropy <br />
window_size(length of input characters): 100 <br />

