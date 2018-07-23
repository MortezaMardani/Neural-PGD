# Neural-PGD
This code implements the neural proximal gradient descent (PGD) algorithm proposed in https://arxiv.org/abs/1806.03963. The idea is to unroll the proximal gradient descent algorithm and model the proximal using a neural network. Adopting residual network (ResNet) as the proximal, a recurrent neural net (RNN) is implemented to learn the proximal. The code is flexible to incorporate a combination of various training costs such as pixel-wise l1/l2, adversarial GAN, LSGAN, and WGAN. 

# Command line

python3 npgd_main.py                 
--run train            
--dataset_train /path/to/train/dataset            
--dataset_test /path/to/test/dataset           
--sampling_pattern /path/to/sampling/trajectory/.matfile           
--sample_size_x 160      
--sample_size_y 128          
--batch_size 2        
--summary_period 20000           
--sample_test -1               
--sample_train -1               
--subsample_test 1000               
--subsample_train 1000               
--train_time 3000               
--train_dir /path/to/save/results              
--checkpoint_dir /path/to/save/checkpoints             
--tensorboard_dir /path/to/save/tensorboard           
--gpu_memory_fraction 1.0               
--hybrid_disc 0            
--starting_batch 0            


# Datasets

# MRI
For medical image reconstruction we adopt the MRI datasets available at the https://www.mridata.org made available as a result of a joint collaboration between Stanford & UC Berkeley. It includes a 20 3D Knee images that have a high resoltuion of 192x320x256. 192 2D axial slices are collected from all patients to form the training and test datasets. 

-- The input files have .jpg format in the train and test folders               
-- The sampling mask is randomly generated based on a avariable density with radial view ordering sampling technique. The        Matlab code is avialble at http://mrsrl.stanford.edu/~jycheng/software.html

# CelebA Face dataset
Adopting celebFaces Attributes Dataset (CelebA) for train and test we use 10K and 1,280 images, respectively. Ground-truth images has 128×128 pixels that is downsampled to 64 × 64 LR images.
