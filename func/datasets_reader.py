from func.utils import *


def batch_read_npyfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test = "train",
                       model_dim = [70, 70],
                       classes = 1):
    '''
    Batch read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param model_dim:               Dimension of one velocity model
    :param classes:                 Number of output channels
    :return:                        a pair: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
    '''

    dataset = None
    labelset = None

    for i in range(start, start + batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))

        if i == start:
            dataset = np.load(filename_seis)
        else:
            dataset = np.append(dataset, np.load(filename_seis), axis=0)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))

        if i == start:
            labelset = np.load(filename_label)
        else:
            labelset = np.append(labelset, np.load(filename_label), axis=0)

    print("Generating velocity model profile......")
    conlabels = np.zeros([batch_length * 500, classes, model_dim[0], model_dim[1]])
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            conlabels[i, j, ...] = extract_contours(labelset[i, j, ...])

    #return dataset, [labelset, conlabels]
    return dataset, labelset

def single_read_npyfile(dataset_dir,
                        readIDs,
                        train_or_test = "train"):
    '''
    Single read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param readID:                  The IDs number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        seismic data, velocity model, contour of velocity model
    '''

    # Determine the seismic data path in the dataset
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_seis))
    # Determine the velocity model path in the dataset
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_label))

    se_data = np.load(filename_seis)[readIDs[1]]
    vm_data = np.load(filename_label)[readIDs[1]][0]

    print("Generating velocity model profile......")
    conlabel = extract_contours(vm_data)

    return se_data, vm_data, conlabel
