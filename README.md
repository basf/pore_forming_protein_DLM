The program was written by Theju Jacob as part of the work outlined in the paper: https://doi.org/10.1038/s41598-022-05970-w

This folder contains scripts and data associated with the deep learning model that was run for the paper. 

The python script is commented, and has `sys.exit()` commands to exit at various points. You will need to uncomment/comment out the `sys.exit()` commands depending on where you want to stop the program. The paths and names of files are hard coded.

The Alignments folder has the training data, including all training and negative data sets. 

Positive training data sets can be found in respective folders. They were downloaded from Uniprot. No internal data is used for training or testing.

Details on how the models works and what it does can be found in our paper:

> Jacob, T., Kahn, T.W. A deep learning model to detect novel pore-forming proteins. Sci Rep 12, 2013 (2022). https://doi.org/10.1038/s41598-022-05970-w

The log file prints out the proteins that were detected as either alpha or beta pore formers. Negatives are not printed out. 
An array of 3 probabilities are also printed -- the first one indicates the probability of being a negative, the second one, the probability of being alpha pore former, and the last one, the probability of being a beta pore former. The program now calls any protein with a probability of > 0.5, as a hit. The threshold can be adjusted in the program. For example,

`Alpha  [0.12113681 0.7647947  0.11406841] >Vip3Aa62`

The above line from the log file indicates that Vip3Aa62 has a probability of 0.7647947 of being an alpha pore former, and the program therefore has chosen to designate it as such.
