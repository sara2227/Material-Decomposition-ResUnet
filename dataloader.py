import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import array
import glob
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        print(image_dir)
        self.image_dir = image_dir
        # self.image_filenames = [item.split('/')[-1] for item in glob.glob(image_dir+'pe_pvc/*_PePvc_pe.raw')] 
        self.transform = ToTensor()

    def __len__(self):
        return int((45-20)*24)

    def __getitem__(self, index):
        output_images = []
        input_images = []
        for i in range(20,46):
            for j in range(1,25):
                
                output_images = np.zeros([3,1024,1024])
                input_images = np.zeros([6,1024,1024])
                
                base_name = 'pht'+str(i)+'pl'+str(j)
                
                output_images[0] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'vox_phantoms/'+base_name+'_vox/'+base_name+'.density_1', dtype='float32').reshape(1024,1024),k=-1)))
                output_images[1] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'vox_phantoms/'+base_name+'_vox/'+base_name+'.density_2', dtype='float32').reshape(1024,1024),k=-1)))
                output_images[2] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'vox_phantoms/'+base_name+'_vox/'+base_name+'.density_3', dtype='float32').reshape(1024,1024),k=-1)))

                input_images[0] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'pe_io/'+base_name+'_PeIo_io.raw', dtype='float32').reshape(1024,1024),k=-1)))
                input_images[1] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'pe_io/'+base_name+'_PeIo_pe.raw', dtype='float32').reshape(1024,1024),k=-1)))
                input_images[2] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'pe_pvc/'+base_name+'_PePvc_pe.raw', dtype='float32').reshape(1024,1024),k=-1)))
                input_images[3] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'pe_pvc/'+base_name+'_PePvc_pvc.raw', dtype='float32').reshape(1024,1024),k=-1)))
                input_images[4] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'pvc_io/'+base_name+'_PvcIo_io.raw', dtype='float32').reshape(1024,1024),k=-1)))
                input_images[5] = self.transform(np.fliplr(np.rot90(np.fromfile(self.image_dir+'pvc_io/'+base_name+'_PvcIo_pvc.raw', dtype='float32').reshape(1024,1024),k=-1)))

        return input_images,output_images