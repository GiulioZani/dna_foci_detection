import numpy as np


class Config:

    tmp_save_dir = "/mnt/tmp_detection_model"

    train_num_workers = 6
    test_num_workers = 3

    # train_num_workers=0
    # test_num_workers=0

    hdf5_filename = "/mnt/foci_detection.hdf5"

    model_name = "detection_model_1"

    train_batch_size = 8
    test_batch_size = 3

    # train_batch_size = 4
    # test_batch_size = 2

    #lr_changes_list = np.cumsum([250, 100, 50, 25])
    lr_changes_list = np.cumsum([15, 10,5])
    # lr_changes_list = np.cumsum([100,50,10,5])
    # lr_changes_list = np.cumsum([30,10,5,5])
    max_epochs = lr_changes_list[-1]
    gamma = 0.1
    init_lr = 0.001

    filters = [16, 32, 64, 128]
    # filters = [4, 8, 16, 32]
    input_size = 3
    output_size = 3

    crop_size = [96, 96]
    #crop_size = [248, 248]
    # crop_size = [64,64]
