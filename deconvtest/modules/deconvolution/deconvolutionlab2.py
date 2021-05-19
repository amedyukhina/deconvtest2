import os


def regularized_inverse_filter(lb, data_path, plugin_path):
    command = 'java -jar {} Run -image file {}conv.tif -psf file {}psf.tif ' \
              '-algorithm RIF {} -monitor no -out stack deconv -path {}'.format(plugin_path,
                                                                                data_path,
                                                                                data_path,
                                                                                lb,
                                                                                data_path)
    print(command)
    os.system(command)
