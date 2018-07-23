# Neural-PGD
This code implements the neural proximal gradient descent (PGD) algorithm proposed in https://arxiv.org/abs/1806.03963. The idea is to unroll the proximal gradient descent algorithm and model the proximal using a neural network. Adopting residual network (ResNet) as the proximal, a recurrent neural net (RNN) is implemented to learn the proximal. The code is flexible to incorporate various training costs such as pixel-wise l1/l2 cost, adversarial GAN, LSGAN, and WGAN. 

The command to run in the command line:

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


Datasets

We use the datasets available at the \textbf{mridata.org} provided by a joint collaboration between Stanford & UC Berkeley

1) the 
