import numpy as np
import configuration as config
import constants as const
import cv2
import imageio

verbose = True
def make_hot(a,num_classes):
    numdata = np.shape(a)[0]
    rval = np.zeros((numdata, num_classes))
    rval[range(numdata), a] = 1
    return rval


def process_imgs(imgs):
    # batch_size x height x width x 3
    batch_size = imgs.shape[0];
    words = np.zeros((batch_size, const.frame_height, const.frame_width, const.frame_channels))
    if batch_size != const.batch_size:
        print('Something is wrong with batch size',batch_size , const.batch_size)
    for i in range(batch_size):

        im = cv2.resize(imgs[i,:], (const.nominal_width,const.nominal_height))

        rand_crop = np.random.rand();
        rand_rgb_channel = np.random.choice(3);
        y = int(rand_crop * (im.shape[0] - const.frame_height))
        x = int(rand_crop * (im.shape[1] - const.frame_width))
        im = im[y:y + const.frame_height, x:x + const.frame_width, rand_rgb_channel];
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2);

        if (verbose):
            prefix = 'tuple_'
            suffix = '_img';
            cv2.imwrite(config.dump_path + prefix  + '_' + str(i) + suffix + '.png', im)


        words[i,:,:,:] = im;

    return words



def process_frames(frames,ordered=True):
    # batch_size x 6 x height x width x 3
    batch_size = frames.shape[0];
    if batch_size != const.batch_size:
        print('Something is wrong with batch size',batch_size , const.batch_size)
    contexts = np.zeros((batch_size, const.frame_height, const.frame_width, const.context_channels))
    for btch_idx in range(batch_size):
        rand_crop = np.random.rand();


        y = int(rand_crop * (const.nominal_height - const.frame_height))
        x = int(rand_crop * (const.nominal_width - const.frame_width))

        rand_rgb_channel = np.random.choice(3);
        stack_diff = np.zeros((const.frame_height, const.frame_width, const.context_channels))
        if (ordered):
            frames_order = np.arange(const.context_channels + 1)
        else:
            frames_order = np.random.permutation(const.context_channels + 1)

        for i in range(const.context_channels):
            current_frame = frames[btch_idx,frames_order[i],:,:,:];
            current_frame= cv2.resize(current_frame, (const.nominal_width, const.nominal_height))
            current_frame = current_frame[y:y + const.frame_height, x:x + const.frame_width, rand_rgb_channel];

            next_frame  = frames[btch_idx, frames_order[i+1], :, :, :];
            next_frame = cv2.resize(next_frame , (const.nominal_width, const.nominal_height))
            next_frame = next_frame [y:y + const.frame_height, x:x + const.frame_width, rand_rgb_channel];


            stack_diff[:, :, i] = current_frame.astype(np.int32) - next_frame.astype(np.int32);

        if (verbose):
            prefix = 'tuple_'
            suffix = '_pkl'
            images = []
            for j in range(5):
                im = stack_diff[:, :, j];
                im = ((im - np.amin(im)) / np.amax(im) - np.amin(im)) * 255
                im = im.astype(np.uint8)
                images.append(im)
                cv2.imwrite(config.dump_path + prefix + '_' + str(ordered) + '_' + str(
                    j) + suffix + '.png', im)
            for j in range(5):
                images.append(np.zeros((const.frame_height, const.frame_width), dtype=np.uint8))
            imageio.mimsave(
                config.dump_path + prefix  + '_' + str(ordered) + '_' + str(btch_idx) + suffix + '.gif',
                images,
                duration=0.5)


        contexts[btch_idx,:,:,:] =stack_diff;
    return contexts;

def data_prepare(data,supervised=True):
    words = process_imgs(data[0])
    context = process_frames(data[1])
    lbls = make_hot(data[2],config.num_classes);

    return words,context,lbls;

