#=======================#
#	Run Training	#
#=======================#

- In Atis_project file run LSTM_RNN.py with python(version3.6)

#===============================#
#	Code Explanation	#
#===============================#

1. data,utils and metrics folders are obtained by the Theano implementation as they are.
2. The project implementation is in  LSTM_RNN.py file
3. In LSTM_RNN.py
    	-preTrainModel function is for pre training the embedding layer(not used)
	-model function creates the grapgh of the network
	-inference function passes the model prediction from a softmax activation
	-loss function: Softmax Cross Entropy
	-train function minimizes the total loss using Adam optimizer
	-evaluate function evaluates based on F1 score
	- trainingSession function include the training loop for the model
