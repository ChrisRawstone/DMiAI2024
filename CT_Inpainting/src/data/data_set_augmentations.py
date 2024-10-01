from PIL import Image
import numpy as np



#functions in from her in this code will return a list of dictionaries, each dictionary will contain
# keys: 'ct_image', 'mask_image', 'tissue_image'
# values: the corresponding images
# Define the base class
class BaseAugmentation:
    """
    child classes of this class must
    return a list of dictionaries, each dictionary will contain
    keys: 'corrupted', 'mask', 'tissue'
    values: the corresponding images / values     
    """
    def __init__(self):
        pass
        
    def apply_mask_ct(self, ct_image, mask_image):        
        """
        Apply mask to the CT image.
        Only non-zero parts of the mask will be applied.
        """
        # Ensure images are in the same size
        mask_image = mask_image.resize(ct_image.size)

        # get the numpy arrays
        ct_image = np.array(ct_image)
        mask_image = np.array(mask_image)       
        # where mask_image is non-zero, apply the mask that makes ct image zero
        result_image = np.where(mask_image != 0, 0, ct_image)
        # turn the numpy array back to image
        result_image = Image.fromarray(result_image)     
        return result_image

    def augmentation(self):
        raise NotImplementedError("Subclasses should implement this method!")

class flipMaskAug(BaseAugmentation):
    def __init__(self):
        super().__init__()

    def augmentation(self, mask=None, ct=None, tissue=None):
        """
        Returns a total of 4 images, corresponding to four 90-degree rotations of the mask applied to the ct image.
        """
        images = []
        rotations = [0, 90, 180, 270]

        for angle in rotations:
            rotated_mask = mask.rotate(angle)
            corrupted_image = self.apply_mask_ct(ct, rotated_mask)

            images.append({
                'corrupted': corrupted_image,
                'mask': rotated_mask,
                'tissue': tissue, 
            })

        return images
    



    

