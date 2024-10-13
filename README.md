# GRU-Based-Chaotic-Pixel-Encryption
Secure ITS traffic image data with GRU-based chaotic pixel encryption. This project implements a three-phase encryption algorithm combining Gated Recurrent Units (GRU) and Sine-Cosine chaotic maps. Protect sensitive traffic information from theft and misuse during transmission with advanced encryption technology.

1. Problem Statement
Use of encryption techniques to protect traffic-related image data collected by
Intelligent Transport Systems (ITS) from potential adversaries. As these smart
devices capture and transmit traffic data in the form of images, this can be vulnerable
to theft and misuse during communication. To address this issue, we are using
encryption algorithm that combines the Gated Recurrent Unit (GRU) and the SineCosine chaotic map. This encryption approach aims to safeguard transport images
through three key phases: key generation, permutation using chaotic sequences, and
diffusion using the GRU approach.

2. Objectives
The objective is to develop an efficient encryption scheme for securing transport
images that resists against various security attacks.
Sub-Objective 1:
Generating intermediate keys and a seed value for creating chaotic sequences from a
combination of 128-bit shared keys and a 128-bit initial vector. To enhance the
security of the encryption process, we will be using the Sine-Cosine chaotic map.


Sub-Objective 2:
Integrating the Gated Recurrent Unit (GRU), a type of recurrent neural network, into
the encryption process. The GRU is used in the diffusion phase, which involves
modifying pixel values in the encrypted images.

3. Methodology
Step 1: Collection of datasets
Step 2: Generating a 128-bit shared key and a 128-bit initial vector
Step 3: Generation of permuted image using sine-cosine map
Step 4: Encryption using Gated Recurrent Unit (GRU)
Step 5: Performance Evaluation using a Bifurcation diagram, Lyapunov exponent and
Shannon entropy

4. Algorithm
Step 1. Load the input image ('in.jpg') using OpenCV.
Step 2. Display the original image.
Step 3. Get the height and width of the image.
Step 4. Generate a chaotic sequence using the `chaoticSequ_Sine_Cosine_Map` function.
This sequence will be used for pixel permutation.
Step 5. Permute the image using the generated chaotic sequence and display the shuffled
image.
Step 6. Define a hidden unit count and calculate the GRU matrix using the chaotic
sequence.
Step 7. Multiply the GRU matrix by a key (key2) to generate an output matrix.
Step 8. Encrypt the shuffled image using the output matrix and key2, displaying the
encrypted image.
Step 9. Decrypt the encrypted image using the same key and output matrix, displaying
the decrypted image.
Step 10. Un shuffle the decrypted image using the same chaotic sequence and display the
unshuffled image.
Step 11. Save the unshuffled image as 'original_row.jpg'.
