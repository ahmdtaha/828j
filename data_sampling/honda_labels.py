honda_labels2num = {'': 0,  ## Background
                        '穿過十字路口 intersection passing': 1,  ## T & + intersection are the same
                        '左轉 left turn': 2,
                        '右轉 right turn': 3,
                        '切左道 left lane change': 4,
                        '切右道 right lane change': 5,
                        '穿過人行道 crosswalk passing': 6,
                        'U轉 U-turn': 7,
                        '左邊分支 left lane branch': 8,
                        '右邊分支 right lane branch': 9,
                        '併道 merge': 10,
                        '穿過T字路口 intersection passing': 1,  ## T & + intersection are the same
                        '穿過鐵路 railroad passing': 0,  ## Ignored as background
                        '駕駛者路旁停車 park': 0,  ## Ignored as background
                        'Park park': 0,  ## Ignored as background
                        'park park': 0}  ## Ignored as background

honda_num2labels = {0: ',穿過鐵路 railroad passing,駕駛者路旁停車 park,Park park,park park',
                    ## Background or railroad passing or parking
                    1: '穿過十字路口 intersection passing,穿過T字路口 intersection passing',
                    2: '左轉 left turn',
                    3: '右轉 right turn',
                    4: '切左道 left lane change',
                    5: '切右道 right lane change',
                    6: '穿過人行道 crosswalk passing',
                    7: 'U轉 U-turn',
                    8: '左邊分支 left lane branch',
                    9: '右邊分支 right lane branch',
                    10: '併道 merge'
                    }

# #print(honda_labels2num)
# print(honda_num2labels[0].split(','))

# import sys
# sys.path.append('../')
# import numpy as np
# import utils
# import pympi
# #
# if __name__ == '__main__':
#     # pkl = utils.pkl_read('/Users/ahmedtaha/Documents/dataset/honda_100h/label_goal.pkl')
#     # print(pkl.keys())
#     # print(pkl['label2num'])
#     # print(type(pkl['num2label']))
#     # sys.exit(1)
#     #files = utils.get_files('/Users/ahmedtaha/Documents/dataset/honda_100h/labels',extension='.pkl',append_base=True);
#     files = utils.get_files('/Users/ahmedtaha/Documents/dataset/honda_100h/event', extension='2017-10-04-11-02-21.eaf', append_base=True);
#
#     layer = u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented'
#     honda_labels = {}
#     i = 0
#
#
#
#     for f in files:
#         eafob = pympi.Elan.Eaf(f)
#         for annotation in eafob.get_annotation_data_for_tier(layer):
#             name = annotation[2].strip()
#
#
#             start = int(np.round(annotation[0] / 1000.)) * 3
#
#             end = int(np.round(annotation[1] / 1000.)) * 3
#             print(start,'\t',end,'\t',name, honda_labels2num[name])
#             # if(name not in honda_labels):
#             #     honda_labels[name] = i;
#             #     i+=1
#             # print(name)
#     #print(honda_labels)
#
#
#     sys.exit(1)
#
#
#
#
#     # honda_labels = {'background': 0,
#     #                 '切右道 right lane change': 7,
#     #                 '左邊分支 left lane branch': 9,
#     #                 'U轉 U-turn': 12,
#     #                 '穿過十字路口 intersection passing': 1,
#     #                 '穿過人行道 crosswalk passing': 11,
#     #                 '右轉 right turn': 3,
#     #                 '併道 merge': 10,
#     #                 '穿過鐵路 railroad passing': 2,
#     #                 '切左道 left lane change': 6,
#     #                 '穿過T字路口 intersection passing': 5,
#     #                 '右邊分支 right lane branch': 8,
#     #                 '左轉 left turn': 4}