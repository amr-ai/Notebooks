#every code runs independently 
                                #Manual Nueral Network [ One input ⇒ One output ]
weight = 0.1
def neural_network(input, weight):
	prediction = input * weight
	return prediction

number_of_toes = [8.5, 9.5, 10, 9]
input = number_of_toes[0]
pred = neural_network(input,weight)
print(pred)

#============================================================================
                            #Manual Nueral Network [ Multiple inputs ⇒ One output ]
def w_sum(a,b):
 assert(len(a) == len(b))
 output = 0
 for i in range(len(a)):
 output += (a[i] * b[i])
 return output

def neural_network(input, weights):
 pred = w_sum(input,weights)
 return pred

weights = [0.1, 0.2, 0]
x = [8.5, 9.5, 9.9, 9.0]
y = [0.65, 0.8, 0.8, 0.9]
z = [1.2, 1.3, 0.5, 1.0]

input = [x[0],y[0],z[0]]
pred = neural_network(input,weights)
print(pred)
#=========================================================================
                #Manual Nueral Network [ Multiple inputs ⇒ One output ] Using Numpy
import numpy as np

def neural_network(input, weights):
	pred = input.dot(weights)
	return pred

weights = np.array([0.1, 0.2, 0])
x= np.array([8.5, 9.5, 9.9, 9.0])
y= np.array([0.65, 0.8, 0.8, 0.9])
z = np.array([1.2, 1.3, 0.5, 1.0])
input = np.array([x[0],y[0],z[0]])

pred = neural_network(input,weights)
print(pred)

#================================================================​
            #Manual Nueral Network [ One input ⇒ Multiple outputs ]
def ele_mul(number,vector):

	output = [0,0,0]
	assert(len(output) == len(vector))
	for i in range(len(vector)):
	output[i] = number * vector[i]
	return output

def neural_network(input, weights):
	pred = ele_mul(input,weights)
	return pred

weights = [0.3, 0.2, 0.9]
x= [0.65, 0.8, 0.8, 0.9]
input = x[0]
pred = neural_network(input,weights)
print(pred)

# =======================================================================================

                #Manual Nueral Network [ One input ⇒ Multiple outputs ]
def w_sum(a,b):
 assert(len(a) == len(b))
 output = 0
 for i in range(len(a)):
 output += (a[i] * b[i])
 return output

def vect_mat_mul(vect,matrix):
 assert(len(vect) == len(matrix))
 output = [0,0,0]
 for i in range(len(vect)):
 output[i] = w_sum(vect,matrix[i])
 return output

def neural_network(input, weights):
 pred = vect_mat_mul(input,weights)
return pred

weights = [ [0.1, 0.1, -0.3],
						[0.1, 0.2, 0.0],
						[0.0, 1.3, 0.1] ]
x= [8.5, 9.5, 9.9, 9.0]
y= [0.65,0.8, 0.8, 0.9]
z= [1.2, 1.3, 0.5, 1.0]

input = [x[0],y[0],z[0]]
pred = neural_network(input,weights)

print(pred)
