'''
# -----------------------------------------
Select Dataset
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # -----------------------------------------------------------
    # FastMRI
    # -----------------------------------------------------------
    if dataset_type in ['fastmri.d.1.0.complex']:
        raise NotImplementedError('Bug here, please use fastmri.d.1.1.complex')
        from data.dataset_FastMRI_complex_d10 import DatasetFastMRI as D

    elif dataset_type in ['fastmri.d.1.1.complex']:
        from data.dataset_FastMRI_complex_d11 import DatasetFastMRI as D

    elif dataset_type in ['fastmri.d.2.0.complex.mc']:
        from data.dataset_FastMRI_complex_mc_d20 import DatasetFastMRI as D

    elif dataset_type in ['fastmri.d.2.0.complex.sc']:
        from data.dataset_FastMRI_complex_sc_d20 import DatasetFastMRI as D

    # -----------------------------------------------------------
    # SKM-TEA
    # -----------------------------------------------------------
    elif dataset_type in ['skmtea.d.0.1.complex']:
        from data.dataset_SKMTEA_complex import DatasetSKMTEA as D



    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
