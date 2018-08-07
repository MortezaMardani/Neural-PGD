import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
import json
from scipy.io import savemat

FLAGS = tf.app.flags.FLAGS
OUTPUT_TRAIN_SAMPLES = 0

def _summarize_progress(train_data, feature, label, gene_output, gene_output_list, eta, nmse, kappa, batch, suffix, max_samples=2, gene_param=None):

    
    td = train_data

    size = [label.shape[1], label.shape[2]]

    # complex input zpad into r and channel
    complex_zpad = tf.image.resize_nearest_neighbor(feature, size)
    complex_zpad = tf.maximum(tf.minimum(complex_zpad, 1.0), 0.0)

    # zpad magnitude
    mag_zpad = tf.sqrt(complex_zpad[:,:,:,0]**2 + complex_zpad[:,:,:,1]**2)
    mag_zpad = tf.maximum(tf.minimum(mag_zpad, 1.0), 0.0)
    mag_zpad = tf.reshape(mag_zpad, [FLAGS.batch_size,size[0],size[1],1])
    mag_zpad = tf.concat(axis=3, values=[mag_zpad, mag_zpad])
    
    # output image
    gene_output_complex = tf.complex(gene_output[:,:,:,0],gene_output[:,:,:,1])
    mag_output = tf.maximum(tf.minimum(tf.abs(gene_output_complex), 1.0), 0.0)
    mag_output = tf.reshape(mag_output, [FLAGS.batch_size, size[0], size[1], 1])
    mag_output = tf.concat(axis=3, values=[mag_output, mag_output])

    label_complex = tf.complex(label[:,:,:,0], label[:,:,:,1])
    label_mag = tf.abs(label_complex)
    label_mag = tf.reshape(label_mag, [FLAGS.batch_size, size[0], size[1], 1])
    mag_gt = tf.concat(axis=3, values=[label_mag, label_mag])

    # concate to visualize image
    image = tf.concat(axis=2, values=[mag_zpad, mag_output, mag_gt])
    image = image[0:FLAGS.batch_size,:,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:,:] for i in range(int(FLAGS.batch_size))])
    image = td.sess.run(image)
    print('save to image size {0} type {1}', image.shape, type(image))
    
    # 3rd channel for visualization
    mag_3rd = np.maximum(image[:,:,0],image[:,:,1])
    image = np.concatenate((image, mag_3rd[:,:,np.newaxis]),axis=2)

    # save to image file
    print('save to image,', image.shape)
    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))


    if gene_param is not None:

        #add feature 
        print('dimension for input, ref, output:',
              feature.shape, label.shape, gene_output.shape)

        gene_param['feature'] = feature.tolist()
        gene_param['label'] = label.tolist()
        gene_param['eta'] = [x.tolist() for x in eta]
        gene_param['nmse'] = [x.tolist() for x in nmse]
        gene_param['kappa'] = [x.tolist() for x in kappa]

        #gene_param['gene_output'] = gene_output.tolist()
        #gene_param['gene_output_save'] = gene_output_save.tolist()
        #add input arguments
        #print(FLAGS.__dict__['__flags'])
        #gene_param['FLAGS'] = FLAGS.__dict__['__flags']

        # save json
        filename = 'batch%06d_%s.json' % (batch, suffix)
        filename = os.path.join(FLAGS.train_dir, filename)
        with open(filename, 'w') as outfile:
            json.dump(gene_param, outfile)
        print("    Saved %s" % (filename,))



def _save_checkpoint(train_data, batch):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver(sharded=True)
    saver.save(td.sess, newname)

    print("Checkpoint saved")



def train_model(train_data, num_sample_train=1984, num_sample_test=116):
    
    td = train_data
    summary_op = td.summary_op

    # update merge_all_summaries() to tf.summary.merge_all
    # summaries = tf.summary.merge_all()
    # td.sess.run(tf.initialize_all_variables()) 
    # td.sess.run(tf.global_variables_initializer())

    #load data

    lrval = FLAGS.learning_rate_start
    start_time = time.time()
    done = False
    batch = FLAGS.starting_batch

    # batch info    
    batch_size = FLAGS.batch_size
    num_batch_train = num_sample_train / batch_size
    num_batch_test = num_sample_test / batch_size            

    # learning rate
    assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)    
    # update: get all test features

    list_test_features = []
    list_test_labels = []
    for batch_test in range(int(num_batch_test)):
        test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
        list_test_features.append(test_feature)
        list_test_labels.append(test_label)
    print('prepare {0} test feature batches'.format(num_batch_test))

    accumuated_err_loss=[]
 

    #tensorboard summary writer

    sum_writer=tf.summary.FileWriter(FLAGS.tensorboard_dir, td.sess.graph)

    while not done:

        batch += 1
        gene_fool_loss = gene_dc_loss = gene_loss = gene_mse_loss = disc_real_loss = disc_fake_loss = -1.234

        #first train based on MSE and then GAN
        if batch < 1e3:
           feed_dict = {td.learning_rate : lrval, td.gene_mse_factor : 1.0}
        else:
           feed_dict = {td.learning_rate : lrval, td.gene_mse_factor : 1/np.sqrt(batch+4-1e3) + 0.5}
        

        ops = [td.gene_minimize, td.disc_minimize, summary_op, td.gene_loss, td.gene_mse_loss, td.gene_fool_loss, td.gene_dc_loss, td.disc_real_loss, td.disc_fake_loss, td.list_gene_losses]                   
        _, _, fet_sum, gene_loss, gene_mse_loss, gene_fool_loss, gene_dc_loss, disc_real_loss, disc_fake_loss, list_gene_losses = td.sess.run(ops, feed_dict=feed_dict)
        
        sum_writer.add_summary(fet_sum,batch)


        # get all losses
        list_gene_losses = [float(x) for x in list_gene_losses]
    
        # verbose training progress
        if batch % 10 == 0:

            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            err_log = 'Progress[{0:3f}%], ETA[{1:4f}m], Batch [{2:4f}], G_MSE_Loss[{3}], G_DC_Loss[{4:5f}], G_Fool_Loss[{5:3.3f}], D_Real_Loss[{6:3.3f}], D_Fake_Loss[{7:3.3f}]'.format(
                    int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, 
                    gene_mse_loss, gene_dc_loss, gene_fool_loss, disc_real_loss, disc_fake_loss)

            print(err_log)

            # update err loss
            err_loss = [int(batch), float(gene_loss), float(gene_dc_loss), 
                        float(gene_fool_loss), float(disc_real_loss), float(disc_fake_loss)]
            accumuated_err_loss.append(err_loss)

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if (current_progress >= 1.0) or (batch > FLAGS.train_time*200):
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        # export test batches
        if batch % FLAGS.summary_period == 0:

            # loop different test batch
            for index_batch_test in range(int(num_batch_test)):

                # get test feature
                test_feature = list_test_features[index_batch_test]
                test_label = list_test_labels[index_batch_test]
            
                # Show progress with test features
                feed_dict = {td.gene_minput: test_feature, td.label_minput: test_label}
                
                ops = [td.gene_moutput, td.gene_moutput_list, td.gene_mlayers_list, td.gene_mask_list, td.gene_mask_list_0, td.disc_layers, td.eta, td.nmse, td.kappa]
                
                # get timing
                forward_passing_time = time.time()
                gene_output, gene_output_list, gene_layers_list, gene_mask_list, gene_mask_list_0, disc_layers, eta, nmse, kappa= td.sess.run(ops, feed_dict=feed_dict)       
                inference_time = time.time() - forward_passing_time


                # save record
                gene_param = {'train_log':err_log,
                              'train_loss':accumuated_err_loss,
                              'gene_loss':list_gene_losses,
                              'inference_time':inference_time,
                              'gene_output_list':[x.tolist() for x in gene_output_list], 
                              'gene_mask_list':[x.tolist() for x in gene_mask_list],
                              'gene_mask_list_0':[x.tolist() for x in gene_mask_list_0]} #,
                              #'gene_mlayers_list':[x.tolist() for x in gene_layers_list]} 
                              #'disc_layers':[x.tolist() for x in disc_layers]}                
                
                # gene layers are too large
                #if index_batch_test>0:
                    #gene_param['gene_layers']=[]

                _summarize_progress(td, test_feature, test_label, gene_output, gene_output_list, eta, nmse, kappa, batch,  
                                    'test{0}'.format(index_batch_test),                                     
                                    max_samples = FLAGS.batch_size,
                                    gene_param = gene_param)
                # try to reduce mem
                gene_output = None
                gene_layers = None
                disc_layers = None
                accumuated_err_loss = []


        # export train batches
        if OUTPUT_TRAIN_SAMPLES and (batch % FLAGS.summary_train_period == 0):

            # get train data
            ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.gene_fool_loss, td.gene_dc_loss, td.disc_real_loss, td.disc_fake_loss, 
                   td.train_features, td.train_labels, td.gene_output]#, td.gene_var_list, td.gene_layers]
            _, _, gene_loss, gene_dc_loss, gene_fool_loss, disc_real_loss, disc_fake_loss, train_feature, train_label, train_output, mask = td.sess.run(ops, feed_dict=feed_dict)
            print('train sample size:',train_feature.shape, train_label.shape, train_output.shape)
            _summarize_progress(td, train_feature, train_label, train_output, batch%num_batch_train, 'train')

        
        # export check points
        if batch % FLAGS.checkpoint_period == 0:
            
            # Save checkpoint
            _save_checkpoint(td, batch)

    _save_checkpoint(td, batch)
    print('Finished training!')

