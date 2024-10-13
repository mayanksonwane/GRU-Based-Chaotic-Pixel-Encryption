import cv2
import numpy as np
import matplotlib.pyplot as plt

def chaoticSequ_Sine_Cosine_Map(x, r, n,h):
    index = []
    k = []

    for j in range(h):
      row = []
      row2=[]
      l=0
      for i in range(n):
        x = np.abs(np.abs(np.sin(-r * x + x*3 - r * np.sin(x))) - np.abs(np.cos(-r * x + x*3 - r * np.sin(x))))  # logistic map
        #print("x->",x)
        x=(x*1000)%n
        row.append(int(x))
        row2.append(l)
        l=l+1
      for i in range(n):
        for pp in range(n):
          if row[i]==row2[pp]:
            row2[pp]=n+2
            break
      row = set(row)
      row = list(row)
      row3=[]
      for i in range(n):
        if row2[i]!=(n+2):
          row3.append(row2[i])

      row=row3 + row
      k.append(row) # Generating key
      #print("x->",k[j][i])
      index.append(row)
       # generating index

    for l in range(h):  # Iterate over every second element in k
      for i in range(n):
          for j in range(n):

              if k[l][i] > k[l][j]:
                  # rearrange key in ascending order
                  k[l][i], k[l][j] = k[l][j], k[l][i]
                  index[l][i], index[l][j] = index[l][j], index[l][i]
    return index

def permutatedImage(img, index, x, y):
    ecrimg = np.zeros(shape=[x, y, 3], dtype=np.uint8)  # Assuming the image is a color image, change to [x, y] if grayscale

    # Convert img to NumPy array
    img_np = np.array(img)

    for i in range(x):
        for j in range(y):
            ecrimg[i][j] = img_np[i][index[i][j]]


    return ecrimg


def GRU(index,h,x,y):
    grumatrix = np.zeros(shape=[x, y], dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            z_inner=(Wz*index[i][j])+(Uz*h_t_demo)+(bz)
            update_gate=torch.sigmoid(z_inner)
            r_inner=(Wr*index[i][j])+(Ur*h_t_demo)+(br)
            reset_gate=torch.sigmoid(r_inner)
            c_init=torch.logical_not(torch.logical_xor(reset_gate, h_t_demo))
            c_inner=(Wh*index[i][j])+(Uh*c_init)+bh
            candidate_activation = torch.tanh(c_inner)
            h_ini=(1-update_gate)
            h_one=torch.logical_not(torch.logical_xor(h_ini, h_t_demo))
            h_two=torch.logical_not(torch.logical_xor(update_gate, candidate_activation))
            grumatrix[i][j] =h_one+h_two

    return grumatrix

def encryption(ecrimg_row, output, key_row, key2, x, y):
    cyimg = np.zeros(shape=[x, y, 3], dtype=np.uint8)  # Assuming the image is a color image, change to [x, y] if grayscale

    for i in range(x):
        for j in range(y):
            cyimg[i][j] = np.bitwise_xor(ecrimg_row[i][j], output[i][j])
            cyimg[i][j] = np.bitwise_xor(cyimg[i][j], np.multiply(key_row[i][j], key2))

    return cyimg


def decryption(ecrimg_row,output,key_row,key2,x, y):
    dyimg= np.zeros(shape=[x, y,3], dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            dyimg[i][j] = np.bitwise_xor(output[i][j],ecrimg_row[i][j])
            dyimg[i][j]=np.bitwise_xor( np.multiply(key_row[i][j], key2) , dyimg[i][j])
            #print(dyimg[i][j])

    return dyimg



def unshuffleimg(decryp_img, index, x, y):
    original_img = np.zeros(shape=[x, y,3], dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            original_img[i][index[i][j]] = decryp_img[i][j]

    return original_img

if _name_ == '_main_':
    img = cv2.imread('in4.jpg')
    plt.imshow(img)
    plt.show()

    height = img.shape[0]  # number of rows
    width = img.shape[1]  # number of columns
    print(height)
    print(width)


    # Shuffling the pixels column-wise
    ChaoticSequence = chaoticSequ_Sine_Cosine_Map(2, 2.11, width,height)
    print(ChaoticSequence)
    PermutatedImage = permutatedImage(img, ChaoticSequence, height, width)
    print(PermutatedImage)




    # Display the shuffled images
    plt.imshow(PermutatedImage)
    plt.show()

    hiddenUnit=10
    GRUVal=GRU(ChaoticSequence,hiddenUnit, height, width)
   # print(GRUVal)

    key2=300
    output=(GRUVal*key2)
    print(output)

    Encryptimage=encryption(PermutatedImage,output,ChaoticSequence,key2,height, width)
    plt.imshow(Encryptimage)
    plt.show()

    decryptimage=decryption(Encryptimage,output,ChaoticSequence,key2,height, width)
    plt.imshow(decryptimage)
    plt.show()


    # Unshuffle the images using the same keys
    original_row = unshuffleimg(decryptimage, ChaoticSequence, height, width)

    # Display the unshuffled images
    plt.imshow(original_row)
    plt.show()

    # Save the unshuffled images
    cv2.imwrite('original_row.jpg', original_row)
