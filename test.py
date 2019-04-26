import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import time
import datetime
import cv2

import Nets
from Data_utils import data_reader, weights_utils, preprocessing
from Losses import loss_factory

def main(args):
    # setup input pipelines
    with tf.variable_scope('input_readers'):

        data_set = data_reader.dataset(
            args.sequence,
            batch_size = 1,
            crop_shape=args.imageSize,
            num_epochs=1,
            augment=False,
            is_training=False,
            shuffle=False
        )
        left_img_batch, right_img_batch, gt_image_batch = data_set.get_batch()

    # build model
    with tf.variable_scope('model'):
        net_args = {}
        net_args['left_img'] = left_img_batch
        net_args['right_img'] = right_img_batch
        net_args['is_training'] = False
        stereo_net = Nets.factory.getStereoNet(args.modelName, net_args)
        print('Stereo Prediction Model:\n', stereo_net)

        # retrieve full resolution prediction and set its shape
        predictions = stereo_net.get_disparities()
        full_res_disp = predictions[-1]
        full_res_shape = left_img_batch.get_shape().as_list()
        full_res_shape[-1] = 1
        full_res_disp.set_shape(full_res_shape)

        # cast img batch to float32 for further elaboration
        right_input = tf.cast(right_img_batch, tf.float32)
        left_input = tf.cast(left_img_batch, tf.float32)
        gt_input = tf.cast(gt_image_batch, tf.float32)

        inputs={}
        inputs['left'] = left_input
        inputs['right'] = right_input
        inputs['target'] = gt_input


        if args.mode != 'SAD':
            reprojection_error = loss_factory.get_reprojection_loss('ssim_l1',reduced=False)([full_res_disp],inputs)[0]
            if args.mode=='WAD':
                weight,_ = Nets.sharedLayers.weighting_network(tf.stop_gradient(reprojection_error),reuse=False)
                adaptation_loss = tf.reduce_sum(reprojection_error*weight)
                if args.summary>1:
                    masked_loss = reprojection_error*weight
                    tf.summary.image('weight',preprocessing.colorize_img(weight,cmap='magma'))
                    tf.summary.image('reprojection_error',preprocessing.colorize_img(reprojection_error,cmap='magma'))
                    tf.summary.image('rescaled_error',preprocessing.colorize_img(masked_loss,cmap='magma'))
            else:	
                adaptation_loss = tf.reduce_mean(reprojection_error)
        else:
            adaptation_loss = loss_factory.get_supervised_loss('mean_l1')([full_res_disp],inputs)

    with tf.variable_scope('validation_error'):
        # get the proper gt
        gt_input = tf.where(tf.is_finite(gt_input),gt_input,tf.zeros_like(gt_input))

        # compute error against gt
        abs_err = tf.abs(full_res_disp - gt_input)
        valid_map = tf.cast(tf.logical_not(tf.equal(gt_input, 0)), tf.float32)
        filtered_error = abs_err * valid_map
        
        if args.summary>1:
            tf.summary.image('filtered_error', filtered_error)

        abs_err = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
        if args.kittiEval:
            error_pixels = tf.math.logical_and(tf.greater(filtered_error, args.badTH),tf.greater(filtered_error, gt_input*0.05))
        else:
            error_pixels = tf.greater(filtered_error, args.badTH)
        bad_pixel_abs = tf.cast(error_pixels,tf.float32)
        bad_pixel_perc = tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map)

        # add summary for epe and bad3
        tf.summary.scalar('EPE', abs_err)
        tf.summary.scalar('bad{}'.format(args.badTH), bad_pixel_perc)

    # setup optimizer and trainign ops
    num_steps = len(data_set)
    with tf.variable_scope('trainer'):	
        if args.mode == 'NONE':
            trainable_variables = []
        else:
            trainable_variables = stereo_net.get_trainable_variables()

        if len(trainable_variables) > 0:
            print('Going to train on {}'.format(len(trainable_variables)))        
            optimizer = tf.train.AdamOptimizer(args.lr)
            train_op = optimizer.minimize(adaptation_loss,var_list=trainable_variables)
        else:
            print('Nothing to train, switching to pure forward')
            train_op = tf.no_op()

    # setup loggin info
    tf.summary.scalar("adaptation_loss", adaptation_loss)

    if args.summary>1:
        tf.summary.image('ground_truth', preprocessing.colorize_img(gt_image_batch,cmap='jet'))
        tf.summary.image('prediction',preprocessing.colorize_img(full_res_disp,cmap='jet'))
        tf.summary.image('left', left_img_batch)

    summary_op = tf.summary.merge_all()

    # create saver and writer to save ckpt and log files
    logger = tf.summary.FileWriter(args.output)

    # adapt
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # init everything
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        # restore weights
        restored, _ = weights_utils.check_for_weights_or_restore_them(args.output, sess, initial_weights=args.weights, prefix=args.prefix, ignore_list=['train_model/'])
        print('Restored weights {}, initial step: {}'.format(restored, 0))
        
        bad3s=[]
        epes=[]
        global_start_time = time.time()
        start_time = time.time()
        step = 0
        try:
            if args.summary>0:
                fetches = [train_op, full_res_disp, adaptation_loss, abs_err, bad_pixel_perc, summary_op]
            else:
                fetches = [train_op, full_res_disp, adaptation_loss, abs_err, bad_pixel_perc, left_img_batch]

            while True:
                # train
                if args.summary>0:
                    _, dispy, lossy, current_epe, current_bad3, summary_string = sess.run(fetches)
                else:
                    _, dispy, lossy, current_epe, current_bad3, lefty = sess.run(fetches)

                
                epes.append(current_epe)
                bad3s.append(current_bad3) 
                if step % 100 == 0:
                    end_time = time.time()
                    elapsed_time = end_time-start_time
                    missing_time = ((num_steps-step)//100) * elapsed_time
                    missing_epochs = 1-(step/num_steps)
                    print('Step:{}\tLoss:{:.2}\tf/b-time:{:.3}s\tmissing time: {}\tmissing epochs: {:.3}'.format(step,lossy, elapsed_time/100, datetime.timedelta(seconds=missing_time), missing_epochs))
                    if args.summary>0:
                        logger.add_summary(summary_string, step)
                    start_time = time.time()

                if args.logDispStep != -1 and step % args.logDispStep == 0:
                    dispy_to_save = np.clip(dispy[0].astype(np.uint16), 0, 256)
                    cv2.imwrite(os.path.join(args.output, 'disparities/disparity_{}.png'.format(step)), dispy_to_save*256)
                    cv2.imwrite(os.path.join(args.output, 'rgbs/left_{}.png'.format(step)), lefty[0,:,:,::-1].astype(np.uint8))

                step += 1
        except tf.errors.OutOfRangeError:
            pass
        finally:
            global_end_time = time.time()
            avg_execution_time = (global_end_time-global_start_time)/step
            fps = 1.0/avg_execution_time

            with open(os.path.join(args.output,'stats.csv'),'w+') as f_out:
                bad3_accumulator = np.sum(bad3s)
                epe_accumulator = np.sum(epes)
                # report series
                f_out.write('AVG_bad{},{}\n'.format(args.badTH,bad3_accumulator/num_steps))
                f_out.write('AVG_EPE,{}\n'.format(epe_accumulator/num_steps))
                f_out.write('AVG Execution time,{}\n'.format(avg_execution_time))
                f_out.write('FPS,{}'.format(fps))
            
            files = [x[0] for x in data_set.get_couples()]
            with open(os.path.join(args.output,'series.csv'),'w+') as f_out:
                f_out.write('Iteration,file,EPE,bad{}\n'.format(args.badTH))
                for i,(f,e,b) in enumerate(zip(files,epes,bad3s)):
                    f_out.write('{},{},{},{}\n'.format(i,f,e,b))
                    
            print('All done shutting down')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to adapt online a Stereo network")
    parser.add_argument("--sequence", required=True, type=str, help='path to the sequence file')
    parser.add_argument("-o", "--output", type=str, help='path to the output folder where stuff will be saved', required=True)
    parser.add_argument("--weights", help="intial weight for the network", default=None)
    parser.add_argument("--modelName", help="name of the stereo model to be used", default="Dispnet", choices=Nets.factory.getAvailableNets())
    parser.add_argument("--lr", help="value for learning rate",default=0.0001, type=float)
    parser.add_argument("--logDispStep", help="save disparity at step multiple of this, -1 to disable saving", default=-1, type=int)
    parser.add_argument("--prefix", help='prefix to be added to the saved variables to restore them', default='')
    parser.add_argument('-m', "--mode", help='choose the adaptation mode, AD to perform standard adaptation, WAD to perform confidence weighted adaptation, NONE to perform just inference', choices=['AD', 'WAD', 'SAD', 'NONE'], required=True)
    parser.add_argument("--summary",help="type of tensorboard summaries: 0 disabled, 1 scalar, 2 scalar+image",type=int, default=0, choices=[0,1,2])
    parser.add_argument("--imageSize", type=int, default=[320,1216], help='two int refering to input image height e width', nargs='+')
    parser.add_argument("--badTH", type=int, default=3, help="threshold for percentage of wrong pixels")
    parser.add_argument("--kittiEval", help="evaluation using kitti2015 protocol: error < badth or lower than 5 percent", action='store_true')
    args = parser.parse_args()
    
    # check image shape
    try:
        assert(len(args.imageSize)==2)
    except Exception as e:
        print('ERROR: invalid image size')
        print(e)
        exit()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.logDispStep!=-1 and not (os.path.exists(os.path.join(args.output, 'disparities')) and os.path.exists(os.path.join(args.output, 'rgbs'))):
        os.makedirs(os.path.join(args.output, 'disparities'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'rgbs'), exist_ok=True)
    with open(os.path.join(args.output, 'params.sh'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')
    main(args)
