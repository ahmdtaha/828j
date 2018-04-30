import logging
import os
import configuration as config


root_logger = logging.getLogger('root_logger')
root_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(config.model_save_path ,'debug.txt'),mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
root_logger.addHandler(fh)


root_logger.info('=========================================================')
root_logger.info('DB? ' + str(config.dataset_name))
root_logger.info('DB-Split? ' + str(config.db_split))
root_logger.info('Reduce Augmentation '+str(config.reduce_overfit))
root_logger.info('Two Stream Model? ' + str(config.use_two_stream))
root_logger.info('=========================================================')
