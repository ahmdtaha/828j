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
