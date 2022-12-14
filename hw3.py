from ast import Lambda
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    dataset = np.load(filename)
    arr = np.array(dataset)
    return arr - np.mean(arr, axis=0)

    # pass

def get_covariance(dataset):
    # Your implementation goes here!
    arr = np.array(dataset)
    
    return np.dot(np.transpose(arr),arr)/(len(dataset)-1)

def get_eig(S, m):
    # Your implementation goes here!
    n = len(S)
    arr = np.array(S)
    E,V = eigh(arr, subset_by_index=[n-m, n-1])
    return np.flip(np.diag(E)), np.flip(V,axis=1)

    

def get_eig_prop(S, prop):
    # Your implementation goes here!
    E = eigh(S, eigvals_only=True)

    sum = 0
    for x in E:
        sum += x
    retE, retV = eigh(S, subset_by_value=(sum * prop,sum))
    return np.flip(np.diag(retE)), np.flip(retV,axis=1)

def project_image(image, U):
    # Your implementation goes here!
    m = len(U)

    ret = np.dot(np.dot(U, np.transpose(U)), image)
    return ret

def display_image(orig, proj):
    # Your implementation goes here!
    orig = np.transpose(np.reshape(orig, newshape=(32,32)))
    proj = np.transpose(np.reshape(proj, newshape=(32,32)))
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    pos1 = ax1.imshow(orig, aspect='equal')
    pos2 = ax2.imshow(proj, aspect='equal')
    fig.colorbar(pos1, ax=ax1)
    fig.colorbar(pos2, ax=ax2)
    plt.show()
    
    pass
x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
Lambda, U = get_eig_prop(S, 0.07)
projection = project_image(x[0],U)
display_image(x[0], projection)
