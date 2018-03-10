import configuration as config
import scipy.io as spio

class O3NWeights:
    def _check_keys(self,dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self,matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict

    def __init__(self):
        self.weight_dict ={};
        mat_weights = spio.loadmat(config.o3n_weights_path, struct_as_record=False, squeeze_me=True);
        mat_weights_dict = self._check_keys(mat_weights)
        print(mat_weights_dict.keys())
        print(type(mat_weights_dict['layers']))
        for layer in mat_weights_dict['layers']:
            if isinstance(layer, spio.matlab.mio5_params.mat_struct):
                layer_dict = self._todict(layer)
                if 'weights' in layer_dict.keys():
                    layer_name = layer_dict['name'];
                    weight_ary = layer_dict['weights'];
                    if weight_ary.shape[0] >0:
                        ## This layer has weights. Some layers like relu don't have weights
                        if(weight_ary.shape[0] == 2): ## Assume kernel + Bias
                            self.weight_dict[layer_name+'_kernel'] = weight_ary[0]
                            self.weight_dict[layer_name + '_bias'] = weight_ary[1]
                        elif(weight_ary.shape[0] == 1): ## dense layer
                            self.weight_dict[layer_name] = weight_ary[0]
                        else:
                            print('Fix ME')




    def get_layernames(self):
        return self.weight_dict.keys();

    def weight_for_layer(self,layer_name):
        return self.weight_dict[layer_name]

    def get_weights_dict(self):
        return self.weight_dict


if __name__ == '__main__':
    o3n_weights = O3NWeights();
    for key in o3n_weights.get_layernames():
        print(key, o3n_weights.weight_for_layer(key).shape)
    print('Different way *************')
    dict = o3n_weights.get_weights_dict();
    for key in dict.keys():
        print(key, dict[key].shape)