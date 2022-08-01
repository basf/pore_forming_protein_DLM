#!/usr/local/bin/anaconda-python3.5
# coding: utf-8
# Cleaner copy of all of the code
# The current script attempts to build a deep learning pipeline to extract information from protein sequences.
# We attempt to 1) Obtain classification of proteins into active and inactive 2) Identify regions of interest in the protein sequence

import sys,os
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from sklearn import preprocessing
#Necessary for saving files
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import (permutations,product)
print("Initialization of Keras Autoencoder")
os.environ["THEANO_FLAGS"] = 'cxx=/usr/bin/g++'
os.system('module load Keras')
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Input
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, Convolution1D, MaxPooling1D, AveragePooling1D, SpatialDropout1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D, SpatialDropout2D, UpSampling2D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
#---------------------------------------------------#
#----------Function to read in fasta files----------#
#---------------------------------------------------#
def readfasta(file,category):
	protein   = ""
	protein_length = []
	protein_matrix = []
	protein_category = []
	protein_name = []
	for line in file:
		#if category == 0 and len(protein_length) >= 5000:
		#	break
		if line[0] == ">":
			protein_name.append(line)
			if len(protein) < 2000 and len(protein) >= 50:
				protein_category.append(category)#(math.ceil(f/4))
				protein_length.append(len(protein))
				zero_trail = "-"*(2000 - len(protein))
				protein+=zero_trail
				protein_matrix.append(protein)
				protein = ""#[]
				previous_line = line			
			elif len(protein) > 0:
				#print("Outlier",len(protein_name),len(protein_matrix),file,line)
				#print(protein_name)
				protein = ""
				protein_name = protein_name[:-2] + protein_name[-1:]
				#print(protein_name)		
		elif len(protein_name) > 0:
			check_line = line.replace(' ','')
			protein+=check_line.strip('\n')
		else:
			protein = ""
    #Making sure we include the last protein:
	if len(protein) < 2000 and len(protein) >= 50:                    
		protein_category.append(category)#(math.ceil(f/4))
		protein_length.append(len(protein))
		zero_trail = "-"*(2000 - len(protein))
		protein+=zero_trail
		protein_matrix.append(protein)
	else:
		protein_name = protein_name[:-1] 
	#print("Length of protein name and matrix",len(protein_name), len(protein_matrix))
	#for i in range(len(protein_matrix)):
		#print(protein_name[i])
		#print(protein_matrix[i])

	return protein_length, protein_category, protein_name, protein_matrix
#---------------------------------------------------#
#------------ One hot encoding proteins-------------#
#---------------------------------------------------#	
def onehotencode(protein_category, protein_matrix):
	AA = np.array(['-','*','U','O','X','B','Z','J','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(AA)
	#print(integer_encoded)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	#encoder = OneHotEncoder()
	#encoder.fit(AA)
	#print(encoder.categories_)

	#encoder.fit(protein_matrix)
	input_matrix = np.array([[[0 for i in range(2000)] for j in range(28)] for k in range(len(protein_category))])
	for j in range(len(protein_category)):	
		for i in range(len(protein_matrix[j])):
			AA_list = np.array(list(protein_matrix[j][i]))
			integer_encoded = label_encoder.transform(AA_list)
			integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
			onehot_encoded = onehot_encoder.transform(integer_encoded)
			#print("Onehot dimensions",len(onehot_encoded), len(onehot_encoded[0]),j,i)
			input_matrix[j,:,i] = onehot_encoded[0]
			#print("Input Matrix",len(input_matrix), len(input_matrix[0]), len(input_matrix[0][0]))
			#print("Protein Matrix",len(protein_matrix), len(protein_matrix[0]), len(protein_matrix[0][0]))
			# input_matrix = onehot_encoder.transform(protein_matrix[i])
			#print(np.shape(input_matrix))
	return input_matrix	

#---------------------------------------------------#
#------------Function to encode proteins------------#
#---------------------------------------------------#
def encodeproteins(protein_category,protein_matrix):
	AA_encode = {}
	#degenerate AAs - 'B(D)','Z(E)','J(L)'
	AA = ['-','*','U','O','X','B','Z','J','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
	F1 = [0,0,0,0,0,1.05,1.357,1.019,0.591,1.343,1.05,1.357,1.006,0.384,0.336,1.239,1.831,1.019,0.663,0.945,0.189,0.931,1.538,0.228,0.032,1.337,0.595,0.26]
	F2 = [0,0,0,0,0,0.302,1.453,0.987,1.302,0.465,0.302,1.453,0.59,1.652,0.417,0.547,0.561,0.987,1.524,0.828,2.081,0.179,0.055,1.399,0.326,0.279,0.009,0.83]
	F3 = [0,0,0,0,0,3.656,1.477,1.505,0.733,0.862,3.656,1.477,1.891,1.33,1.673,2.131,0.533,1.505,2.219,1.299,1.628,3.005,1.502,4.76,2.213,0.544,0.672,3.097]
	F4 = [0,0,0,0,0,0.259,0.113,1.266,1.57,1.02,0.259,0.113,0.397,1.045,1.474,0.393,0.277,1.266,1.005,0.169,0.421,0.503,0.44,0.67,0.908,1.242,2.128,0.838]
	F5 = [0,0,0,0,0,3.242,0.837,0.912,0.146,0.255,3.242,0.837,0.412,2.064,0.078,0.816,1.648,0.912,1.212,0.933,1.392,1.853,2.897,2.647,1.313,1.262,0.184,1.512]

	#print(len(AA), len(F1), len(F2), len(F3), len(F4), len(F5))
	for i in range(len(AA)):
		AA_encode[AA[i]] = i
		
	input_matrix = [[[0 for i in range(2000)] for j in range(5)] for k in range(len(protein_category))]
	for p in range(len(protein_category)):
		for aa in range(len(protein_matrix[p])):
			key = AA_encode[protein_matrix[p][aa]]
			input_matrix[p][0][aa] = F1[key]
			input_matrix[p][1][aa] = F2[key]
			input_matrix[p][2][aa] = F3[key]
			input_matrix[p][3][aa] = F4[key]
			input_matrix[p][4][aa] = F5[key]
		
	return input_matrix	

#-------------------------------------------------------#
#------------Function to encode proteins - 2------------#
#-------------------------------------------------------#
#Encoding protein sequence trigrams
class ColumnApplier(object):
	def __init__(self, column_stages):
		self._column_stages = column_stages

	def fit(self, X, y):
		for i, k in self._column_stages.items():
			k.fit(X[i])

		return self

	def transform(self, X):
		X = X.copy()
		for i, k in self._column_stages.items():
			X[i] = k.transform(X[i])

		return X
def encodetrigrams(protein_category, protein_matrix):
	AA = '-*UOXBZJACDEFGHIKLMNPQRSTVWY'#'RHKDESTNQCUGPAVILMFYW-'
	n=3
	AA_trigrams = list(map("".join, product(AA, repeat = n)))#product(AA,repeat = 3)#list(map("".join, permutations(AA, n)))#[AA[i:i+n] for i in range(len(AA)-n+1)]
	#print("Combinations generated")
	#print(AA_trigrams)
	protein_matrix_trigrams = []
	for i in range(len(protein_matrix)):
		seq = protein_matrix[i]
		seq_trigrams = [seq[j:j+n] for j in range(len(seq)-n+1)]
		protein_matrix_trigrams.append(seq_trigrams)

	#print("Dimensions of protein_matrix_trigrams")
	#print(len(protein_matrix_trigrams), len(protein_matrix_trigrams[0]), len(protein_matrix_trigrams[1]))

	protein_matrix_trigrams = []
	for i in range(len(protein_matrix)):
		seq = protein_matrix[i]
		seq_trigrams = [seq[j:j+n] for j in range(len(seq)-n+1)]
		protein_matrix_trigrams.append(seq_trigrams)
		

	le = preprocessing.LabelEncoder()
	le.fit(AA_trigrams)
	multi_encoder = ColumnApplier(dict([(i, le) for i in range(len(protein_matrix_trigrams))]))
	#multi_encoder.fit(protein_matrix, None).transform(protein_matrix)
	input_matrix = multi_encoder.transform(protein_matrix_trigrams)
	
	return input_matrix

#---------------------------------------------------#
#-------------------Start of program----------------#
#---------------------------------------------------#	
#Read in fasta files with all toxic proteins
count = 0
protein = ""
protein_length = []
protein_category = []
protein_matrix = []
protein_name = []

#Testing cullpdb as our negative fileset
path = "./"
filename = ["Cullpdb_5616HitsRemoved.fasta"]#["Culledpdb_Nan.fasta"]#["Cullpdb_HitsRemoved.fasta"]
for f in range(len(filename)):
    with open(os.path.join(path,filename[f])) as file:
        count = count + 1
        sub_length, sub_category, sub_name, sub_matrix = readfasta(file, 0)#/3 should be replaced by /1 
        protein_length.extend(sub_length)
        protein_category.extend(sub_category)
        protein_name.extend(sub_name)
        protein_matrix.extend(sub_matrix)
print("Max and no of cullpdb length", max(protein_length), len(protein_length))
path = "../Alignments/"
filename = ["alpha_pfts/Uniprot_Actinoporin_148_cleanv1.fasta","alpha_pfts/Uniprot_Colicin_1000_cleanv1_NoToxin10.fasta","alpha_pfts/Uniprot_Hemolysin_2000_cleanv1_NoToxin10.fasta","alpha_pfts/Uniprot_PesticidalCrystal_334_cleanv1.fasta","beta_pfts/Uniprot_Perfringolysin_AlphaHemolysin_Leucocidin_422_cleanv1.fasta","beta_pfts/Uniprot_Aerolysin_419_cleanv1.fasta","beta_pfts/Uniprot_Cytolysin_2023_cleanv1_NoPerforin.fasta","beta_pfts/Uniprot_Haemolysin_387_cleanv1.fasta"]
#["alpha_pfts/alphapfts_cleanv1_cdhit_70.fasta","beta_pfts/betapfts_cleanv1_cdhit_70.fasta"]#

for f in range(len(filename)):
    #print(math.ceil(f/4))
    with open(os.path.join(path,filename[f])) as file:
        count = count + 1
        sub_length, sub_category, sub_name, sub_matrix = readfasta(file, math.ceil((f+1)/4))#/3 should be replaced by /1 
        protein_length.extend(sub_length)
        protein_category.extend(sub_category)
        protein_name.extend(sub_name)
        protein_matrix.extend(sub_matrix)


print("Dimensions of protein length, category, matrices")		
print(len(protein_length),len(protein_category))
print(len(protein_matrix), len(protein_matrix[0]), len(protein_matrix[0][0]))
#print("Length of proteins, min and max",min(protein_length), max(protein_length))
print("Protein Categories and no:of negatives, alphas, betas:")
#print(protein_category)
print(protein_category.count(0), protein_category.count(1), protein_category.count(2)) 
#sys.exit()

print("Testing with public toxins")
test_protein   = ""#[]
test_protein_length = []
test_protein_matrix = []
test_protein_category = []
test_protein_name = []

path = "../Alignments/testproteins/"
filename = ["Vip3_Public.fasta", "MACPF_Public.fasta", "Toxin10_Public.fasta"]#,"xaaaaa"]
for f in range(len(filename)):
    with open(os.path.join(path,filename[f])) as file:
        sub_length, sub_category, sub_name, sub_matrix = readfasta(file, f)
        test_protein_length.extend(sub_length)
        test_protein_category.extend(sub_category)
        test_protein_name.extend(sub_name)
        test_protein_matrix.extend(sub_matrix)

print("Dimensions of test protein length and category", len(test_protein_length),len(test_protein_category))
print("Min and max of test protein length", min(test_protein_length), max(test_protein_length))

#dim lets you easily switch between encoding methods 
for dim in [5,28,33]:
	if dim == 5:
		print("Gaussian filtering of input data with sigma = 5")
		input_matrix =  encodeproteins(protein_category, protein_matrix)
		sig = 5
		for i in range(len(protein_category)):
			for j in range(dim):#(5):
				input_matrix[i][j] = gaussian_filter(input_matrix[i][j], sigma = sig)
	if dim == 28:
		input_matrix =  onehotencode(protein_category, protein_matrix)
	
	if dim == 64:
		input_matrix = learn_embeddings(protein_category, protein_matrix)

	if dim == 33:
		input_matrix_onehot = onehotencode(protein_category, protein_matrix)
		input_matrix_atchley= encodeproteins(protein_category, protein_matrix) 
		sig = 5
		for i in range(len(protein_category)):
			for j in range(5):#(5):
				input_matrix_atchley[i][j] = gaussian_filter(input_matrix_atchley[i][j], sigma = sig)
		input_matrix = np.concatenate((input_matrix_onehot, input_matrix_atchley), axis=1)
		
	print("Input matrix dimensions")
	print(len(input_matrix), len(input_matrix[0]))#, len(input_matrix[0][0]))

	print(input_matrix[0])
	#sys.exit()
	model = Sequential()

	#Encoding
	# model.add(Conv2D(filters=25, kernel_size=(1,100), activation = 'relu', use_bias=False, input_shape=(5,2000,1) ))#(1,50), (5,100), input_shape=(5,2000,1)))
	# model.add(AveragePooling2D(pool_size=(1,5)))

	# model.add(Conv2D(filters=25, kernel_size=(1,50), activation = 'relu', use_bias=False, input_shape=(5,2000,1) ))#(1,50), (5,100), input_shape=(5,2000,1)))
	# model.add(AveragePooling2D(pool_size=(1,5)))

	if dim != 64:
		model.add(Conv2D(filters=25, kernel_size=(1,100), activation = 'relu', use_bias=False, input_shape=(dim,2000,1) ))#(1,50), (5,100), input_shape=(5,2000,1)))
		model.add(AveragePooling2D(pool_size=(1,5)))

		model.add(Conv2D(filters=25, kernel_size=(1,50), activation = 'relu', use_bias=False, input_shape=(dim,2000,1) ))#(1,50), (5,100), input_shape=(5,2000,1)))
		model.add(AveragePooling2D(pool_size=(1,5)))
		model.add(SpatialDropout2D(0.25))

	else:
		model.add(Conv1D(filters=25, kernel_size=(16), activation = 'relu', use_bias=False, input_shape=(dim,1) ))#(1,50), (5,100), input_shape=(5,2000,1)))
		model.add(AveragePooling1D(pool_size=(4)))

		model.add(Conv1D(filters=25, kernel_size=(4), activation = 'relu', use_bias=False, input_shape=(dim,1) ))#(1,50), (5,100), input_shape=(5,2000,1)))
		model.add(AveragePooling1D(pool_size=(2)))
		model.add(SpatialDropout1D(0.25))
	
	model.add(Flatten())
	#model.add(Dense(125, activation='softmax'))
	model.add(Dense(3, activation='softmax'))

	model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['acc'])
	#summarize the model
	print(model.summary())


	no_prot = len(protein_category)
	#Split into training and testing sets
	train = np.random.choice(no_prot, int(np.floor((no_prot*0.8))))
	test = [i for i in range(no_prot) if i not in train]
	#change 1,2000,1 to 5,2000,1
	if dim == 5:
		atchley_history = model.fit(np.array([input_matrix[j] for j in train]).reshape(len(train),dim,2000,1),to_categorical([protein_category[j] for j in train], 3), validation_split=0.2, shuffle = True, epochs=150, verbose=0)
	if dim == 28:
		onehot_history = model.fit(np.array([input_matrix[j] for j in train]).reshape(len(train),dim,2000,1),to_categorical([protein_category[j] for j in train], 3), validation_split=0.2, shuffle = True, epochs=150, verbose=0)
	if dim == 33:
		ohatchley_history = model.fit(np.array([input_matrix[j] for j in train]).reshape(len(train),dim,2000,1),to_categorical([protein_category[j] for j in train], 3), validation_split=0.2, shuffle = True, epochs=150, verbose=0)
	if dim == 64:
		embedding_history = model.fit(np.array([input_matrix[j] for j in train]).reshape(len(train),dim,1),to_categorical([protein_category[j] for j in train], 3), validation_split=0.2, shuffle = True, epochs=150, verbose=0)

	if dim == 5:
		test_input_matrix =  encodeproteins (test_protein_category, test_protein_matrix)

		for i in range(len(test_protein_category)):
			for j in range(dim):#(5)
				test_input_matrix[i][j] = gaussian_filter(test_input_matrix[i][j], sigma = sig)
	if dim == 28:
		test_input_matrix = onehotencode(test_protein_category, test_protein_matrix)

	if dim == 33:
		test_matrix_onehot = onehotencode(test_protein_category, test_protein_matrix)
		test_matrix_atchley= encodeproteins(test_protein_category, test_protein_matrix) 
		test_input_matrix = np.concatenate((test_matrix_onehot, test_matrix_atchley), axis=1)

	if dim == 64: 
		test_input_matrix = learn_embeddings(test_protein_category, test_protein_matrix)

	print("Test input matrix dimensions")
	print(len(test_input_matrix), len(test_input_matrix[0]))#, len(test_input_matrix[0][0]))

	for k in range(len(filename)):
		print("Testing with proteins from:", filename[k])
		selected = [i for i in range(len(test_protein_category)) if test_protein_category[i] == k]
		if dim != 64:
			selected_output = model.predict(np.array([test_input_matrix[j] for j in selected]).reshape(len(selected),dim,2000,1))
		if dim == 64:
			selected_output = model.predict(np.array([test_input_matrix[j] for j in selected]).reshape(len(selected),dim,1))
		fraction_output = selected_output
		selected_output = np.round(selected_output)
		#print(selected_output)
		selected_summary = [0]*3
		for i in range(len(selected)):
			selected_summary = np.array(selected_summary) + np.array(selected_output[i])
			if k >= 0:
				if selected_output[i][1] == 1:#fraction_output[i][1] >= 0.9: #
					print("Alpha", i, fraction_output[i], test_protein_name[selected[i]])
				if selected_output[i][2] == 1:#fraction_output[i][2] >= 0.9: #
					print("Beta", i, fraction_output[i], test_protein_name[selected[i]])

		print("Dimensions and stats", dim, filename[k],len(selected_output),selected_summary)
		
# summarize history for accuracy
plt.plot(ohatchley_history.history['acc'])
plt.plot(ohatchley_history.history['val_acc'])
plt.plot(atchley_history.history['acc'])
plt.plot(atchley_history.history['val_acc'])
plt.plot(onehot_history.history['acc'])
plt.plot(onehot_history.history['val_acc'])
#plt.plot(embedding_history.history['acc'])
#plt.plot(embedding_history.history['val_acc'])
plt.title('model accuracy')
plt.ylim([0.25, 1.05])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['combined_train', 'combined_test','atchley_train', 'atchley_test','onehot_train', 'onehot_test'], loc='lower right')
plt.savefig('Accuracy_All.png')#('Accuracy_Atchley.png')#('Accuracy_OneHot.png')#('Accuracy_Embeddings.png')#
plt.close()
# summarize history for loss
plt.plot(ohatchley_history.history['loss'])
plt.plot(ohatchley_history.history['val_loss'])
plt.plot(atchley_history.history['loss'])
plt.plot(atchley_history.history['val_loss'])
plt.plot(onehot_history.history['loss'])
plt.plot(onehot_history.history['val_loss'])
#plt.plot(embedding_history.history['loss'])
#plt.plot(embedding_history.history['val_loss'])
plt.title('model loss')
plt.ylim([0.0, 0.35])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['combined_train', 'combined_test','atchley_train', 'atchley_test','onehot_train', 'onehot_test'], loc='upper left')
plt.savefig('Loss_All.png')#('Loss_Atchley.png')#('Loss_OneHot.png')#('Loss_Embeddings.png')#
plt.close()

#------------------------------#
#----------ROC Curves----------#
#------------------------------#
# Compute ROC curve and ROC area for each class
#Reference: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
#y = label_binarize(y, classes=[0, 1, 2])
lw = 2
y_test  = to_categorical([protein_category[j] for j in test])
y_score = model.predict(np.array([input_matrix[j] for j in test]).reshape(len(test),dim,2000,1))
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='orange', linewidth=4)#, linestyle=':', linewidth=4)

#plt.plot(fpr["macro"], tpr["macro"],
         #label='macro-average ROC curve (area = {0:0.2f})'
         #      ''.format(roc_auc["macro"]),
         #color='purple', linewidth=4)#, linestyle=':', linewidth=4)

colors = cycle(['blue','green','red'])#(['aqua', 'darkorange', 'cornflowerblue'])
category = ['non-pft','alpha-pft','beta-pft']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(category[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves of the combined encoding method')
plt.legend(loc="lower right")
plt.savefig("ROC_curves.png")


# # Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot(fpr["micro"], tpr["micro"],
         # label='micro-average ROC curve (area = {0:0.2f})'
               # ''.format(roc_auc["micro"]),
         # color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
         # label='macro-average ROC curve (area = {0:0.2f})'
               # ''.format(roc_auc["macro"]),
         # color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
    # plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             # label='ROC curve of class {0} (area = {1:0.2f})'
             # ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.savefig("2ROC_curves.png")
#sys.exit()
#------------------------------#
