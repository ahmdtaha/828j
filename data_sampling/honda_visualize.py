import sys
sys.path.append('../')
import utils
import data_sampling.honda_labels as honda_lbls
import imageio
import configuration as config
if __name__ == '__main__':
    pkl_path = '/Users/ahmedtaha/Documents/dataset/honda_100h/labels/201710041102_goal.pkl'
    annotations = utils.pkl_read(pkl_path)
    num_events = len(annotations['s'])-1;
    print('num_events ',num_events)
    for event in range(num_events):
        # if(event == 0):
        #     continue;
        print('Start ',annotations['s'][event],' End',annotations['s'][event+1],' Goal:', honda_lbls.honda_num2labels[annotations['G'][event]])

        images = []

        for frame_idx in range(annotations['s'][event],annotations['s'][event+1]):
            frame_path = '/Users/ahmedtaha/Documents/dataset/honda_100h/frames/201710041102/frame_%04d.jpg' % (frame_idx+1);
            im = imageio.imread(frame_path)
            images.append(im)


        imageio.mimsave(config.dump_path + 'event_'+str(event)+'_'+ honda_lbls.honda_num2labels[annotations['G'][event]].split(',')[0] + '.gif',images,duration=0.5)

    #sprint(len())