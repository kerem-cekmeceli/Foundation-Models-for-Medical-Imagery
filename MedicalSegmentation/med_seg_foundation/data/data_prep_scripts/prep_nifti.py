import SimpleITK as sitk


"""
    @TODO
    1) N4 bias corr
    2) bicubic interp
    3) Norm [0, 1]
"""



pth_flair = "/home/kerem_ubuntu/Projects/DataFoundationModels/brain/brats_val/0_FLAIR.nii.gz"
pth_t1 = "/home/kerem_ubuntu/Projects/DataFoundationModels/brain/brats_val/0_T1.nii.gz"
pth_label = "/home/kerem_ubuntu/Projects/DataFoundationModels/brain/brats_val/0_Label.nii.gz"


flair = sitk.GetArrayFromImage(sitk.ReadImage(pth_flair))
t1 = sitk.GetArrayFromImage(sitk.ReadImage(pth_t1))
label = sitk.GetArrayFromImage(sitk.ReadImage(pth_label))

print()