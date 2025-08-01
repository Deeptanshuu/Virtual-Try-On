from io import BytesIO
from PIL import Image
import numpy as np
import cv2

class ImageFormatConvertor:
    # Function to convert PIL Image to Binary Data
    @classmethod
    def pil_image_to_binary_data(cls, pil_image, format='PNG'):
        # Create a buffer to hold the image data
        buffer = BytesIO()
        # Save the PIL image to the buffer in the specified format
        pil_image.save(buffer, format=format)
        # Get the byte data from the buffer
        binary_data = buffer.getvalue()
        return binary_data        
    
    # Function to convert Binary Format to PIL Image
    @classmethod
    def binary_data_to_pil_image(cls, binary_data):
        # Create a BytesIO object from the binary data
        buffer = BytesIO(binary_data)
        # Open the image from the buffer
        pil_image = Image.open(buffer)
        return pil_image
    
    # Function to convert PIL Image to OpenCV format
    @classmethod
    def pil_to_cv2(cls, pil_image):
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR if it's a 3-channel image
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        # Convert RGBA to BGRA if it's a 4-channel image
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
            open_cv_image = open_cv_image[:, :, [2, 1, 0, 3]].copy()
        
        return open_cv_image

    # Function to convert OpenCV format to PIL Image
    @classmethod
    def cv2_to_pil(cls, cv2_image):
        # Convert BGR to RGB if it's a 3-channel image
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert BGRA to RGBA if it's a 4-channel image
        elif len(cv2_image.shape) == 3 and cv2_image.shape[2] == 4:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)            
        pil_image = Image.fromarray(cv2_image)
        return pil_image        