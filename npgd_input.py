import tensorflow as tf
import numpy as np
import math as math

FLAGS = tf.app.flags.FLAGS


'''
# generate mask based on alpha
def generate_mask_alpha(size=[128,128], r_factor_designed=5.0, r_alpha=3, axis_undersample=1,
                        acs=3, seed=0, mute=0):
    # init
    mask = np.zeros(size)
    if seed>=0:
        np.random.seed(seed)

    # get samples
    num_phase_encode = size[axis_undersample]
    num_phase_sampled = int(np.floor(num_phase_encode/r_factor_designed))

    # coordinate
    coordinate_normalized = np.array(range(num_phase_encode))
    coordinate_normalized = np.abs(coordinate_normalized-num_phase_encode/2)/(num_phase_encode/2.0)
    prob_sample = coordinate_normalized**r_alpha
    prob_sample = prob_sample/sum(prob_sample)

    # sample
    index_sample = np.random.choice(num_phase_encode, size=num_phase_sampled, 
                                    replace=False, p=prob_sample)
    # sample                
    if axis_undersample == 0:
        mask[index_sample,:]=1
    else:
        mask[:,index_sample]=1

    # acs                
    if axis_undersample == 0:
        mask[:int((acs+1)/2),:]=1
        mask[-int(acs/2):,:]=1
    else:
        mask[:,:int((acs+1)/2)]=1
        mask[:,-int(acs/2):]=1

    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('gen mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
        print(num_phase_encode, num_phase_sampled, np.where(mask[0,:]))

    return mask, r_factor
'''

# generate mask based on .mat mask
def generate_mask_mat(mask=[], mute=0):

    # shift
    mask = np.fft.ifftshift(mask)

    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('load mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))

    return mask, r_factor


def setup_inputs_one_sources(sess, filenames_input, filenames_output, image_size=None, 
                             axis_undersample=1, capacity_factor=1, 
                             r_factor=4, r_alpha=0, r_seed=0,
                             sampling_mask=None, num_threads=1):

    '''
    # image size
    if image_size is None:
        image_size = [FLAGS.sample_size_x, FLAGS.sample_size_y]

    # generate default mask
    if sampling_mask is None:
        DEFAULT_MASK, _ = generate_mask_alpha(image_size, # kspace size
                                              r_factor_designed=r_factor, 
                                              r_alpha=r_alpha, 
                                              seed=r_seed,
                                              axis_undersample=axis_undersample
                                              )
    else:
        # get input mask
        DEFAULT_MASK, _ = generate_mask_mat(sampling_mask)
    '''

    # get input mask
    DEFAULT_MASK, _ = generate_mask_mat(sampling_mask)

    # convert to complex tf tensor
    DEFAULT_MAKS_TF = tf.cast(tf.constant(DEFAULT_MASK), tf.float32)
    DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)

    
    # Read each JPEG file
    reader_input = tf.WholeFileReader()
    filename_queue_input = tf.train.string_input_producer(filenames_input, shuffle=False)
    key, value_input = reader_input.read(filename_queue_input)
    channels = 3
    image_input = tf.image.decode_jpeg(value_input, channels=channels, name="input_image")
    image_input.set_shape([image_size[0], 2*image_size[1], channels])
    #print('size_input_image', image_input.get_shape())
    image_input = image_input[:,:,-1]   


    '''
    seperate coil sensitivities and images/coil sens are formed as a list
    '''
 
    #choose the complex-valued image
    image_input_mag = tf.cast(image_input[0:image_size[0],0:image_size[1]], tf.complex64)
    image_input_phase = tf.cast(8*tf.constant(math.pi), tf.complex64)*tf.cast(image_input[0:image_size[0],image_size[1]:2*image_size[1]], tf.complex64)
    image_input = tf.multiply(image_input_mag, tf.exp(tf.sqrt(tf.cast(-1,tf.complex64))*image_input_phase))
    image_input = tf.cast(image_input, tf.complex64)
    image_input = image_input / 255.0       #tf.cast(tf.reduce_max(tf.abs(image_input)), tf.complex64)
 
    print('image_input_complex', image_input.get_shape())

    ##choose the magnitude
    #image_input = image_input[0:image_size[0],0:image_size[1]]

    # cast image to float in 0~1
    #image_input = tf.cast(image_input, tf.float32)/255.0

    # output, gold-standard
    image_output = image_input

    # apply undersampling mask
    kspace_input = tf.fft2d(tf.cast(image_input,tf.complex64))
    kspace_zpad = kspace_input * DEFAULT_MAKS_TF_c

    # zpad undersampled image for input
    image_zpad = tf.ifft2d(kspace_zpad)
    image_zpad_real = tf.real(image_zpad)
    image_zpad_real = tf.reshape(image_zpad_real, [image_size[0], image_size[1], 1])
    image_zpad_imag = tf.imag(image_zpad)
    image_zpad_imag = tf.reshape(image_zpad_imag, [image_size[0], image_size[1], 1])    
    image_zpad_concat = tf.concat(axis=2, values=[image_zpad_real, image_zpad_imag])

    # split the complex label into real and imaginary channels
    image_output_real = tf.real(image_output)
    image_output_real = tf.reshape(image_output_real, [image_size[0], image_size[1], 1])
    image_output_complex = tf.imag(image_output)
    image_output_complex = tf.reshape(image_output_complex, [image_size[0], image_size[1], 1])
    image_output_concat = tf.concat(axis=2, values=[image_output_real, image_output_complex])
    

    # The feature is zpad image with 2 channel, label is the ground-truth image with 2 channel
    feature = tf.reshape(image_zpad_concat, [image_size[0], image_size[1], 2])
    label = tf.reshape(image_output_concat, [image_size[0], image_size[1], 2])
    mask = tf.reshape(DEFAULT_MAKS_TF_c, [image_size[0], image_size[1]])

    # Using asynchronous queues
    features, labels, masks = tf.train.batch([feature, label, mask],
                                      batch_size = FLAGS.batch_size,
                                      num_threads = num_threads,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name = 'labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels, masks    
