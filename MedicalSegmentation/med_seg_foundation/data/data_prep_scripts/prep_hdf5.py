import h5py

dataset = 'acdc' # 'nci' , 'acdc', 'rvsc'
cluster = True

if cluster:
    main_pth = "/usr/bmicnas02/data-biwi-01/foundation_models/da_data/"
else:
    main_pth = "/home/kerem_ubuntu/Projects/DataFoundationModels/DataFoundationModels/hdf5/"

# Prostate NCI
if dataset=='nci':
    sub_path = "nci/"  
    filename = 'data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5'

    train_vols = 15
    val_vols = 5
    test_vols = 10

# Cardiac ACDC
elif dataset=='acdc':
    sub_path = "acdc/"  
    filename = 'data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5'

    train_vols = 120
    val_vols = 40
    test_vols = 40

# Cardiac RVSC
elif dataset=='rvsc':
    sub_path = "rvsc/"  
    filename = 'data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5'

    train_vols = 48
    val_vols = 24
    test_vols = 24

else:
    ValueError('unknown dataset')

dir_path = main_pth + sub_path
pth_full = dir_path + filename

# pth = main_pth + 'brain/abide/stanford/'+'data_T1_original_depth_132_from_10_to_15.hdf5'

def get_train_val_test_dicts(pth, train_suff='_train', val_suff='_validation', test_suff='_test'):
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()
    
    with h5py.File(pth, "r") as f:
        print("Keys: %s" % f.keys())
        
        for key in f.keys():
            if train_suff in key:
                key_wo_suffix = key.replace(train_suff, '')
                key_wo_suffix = 'labels' if key_wo_suffix=='masks' else key_wo_suffix
                train_dict[key_wo_suffix] = f[key][:]

            elif val_suff in key:
                key_wo_suffix = key.replace(val_suff, '')
                key_wo_suffix = 'labels' if key_wo_suffix=='masks' else key_wo_suffix
                val_dict[key_wo_suffix] = f[key][:]

            elif test_suff in key:
                key_wo_suffix = key.replace(test_suff, '')
                key_wo_suffix = 'labels' if key_wo_suffix=='masks' else key_wo_suffix
                test_dict[key_wo_suffix] = f[key][:]

            else:
                ValueError(f'Undefined key: {key}')
    
    assert train_dict['nz'].shape[0] == train_vols
    assert val_dict['nz'].shape[0] == val_vols
    assert test_dict['nz'].shape[0] == test_vols
    
    assert train_dict.keys() == val_dict.keys() == test_dict.keys()
    
    return train_dict, val_dict, test_dict

def write_dict_as_hdf5(dct, pth, name):
    if not isinstance(pth, str):
        pth = str(pth)
    hf_file = h5py.File(pth+f'{name}.hdf5', 'w')
    for key, val in dct.items():
        hf_file.create_dataset(key, data=val)
    hf_file.close()
            

def save_dicts_as_hdf5(pth, train_dict=None, val_dict=None, test_dict=None):    
    assert train_dict is not None or val_dict is not None or test_dict is not None
    
    if train_dict is not None:
        print("Writing train files")
        write_dict_as_hdf5(dct=train_dict, pth=pth, name='train')
    
    if val_dict is not None:
        print("Writing val files")
        write_dict_as_hdf5(dct=train_dict, pth=pth, name='val')
    
    if test_dict is not None:
        print("Writing test files")
        write_dict_as_hdf5(dct=train_dict, pth=pth, name='test')
    
    print("Done !")
    

pth = '/usr/bmicnas02/data-biwi-01/foundation_models/da_data/brain/abide/stanford/' 
file_n = 'data_T1_original_depth_132_from_10_to_15.hdf5'
pth += file_n
with h5py.File(pth, "r") as f:
    print("Keys: %s" % f.keys())

# train_dict, val_dict, test_dict = get_train_val_test_dicts(pth=pth_full)

# save_dicts_as_hdf5(pth=dir_path, train_dict=train_dict, val_dict=val_dict, test_dict=test_dict)


pth = dir_path+'test.hdf5'
with h5py.File(pth, "r") as f:

    print("Keys: %s" % f.keys())
    keys = f.keys()
    # print(f[keys[0]].shape)
    
  
print()