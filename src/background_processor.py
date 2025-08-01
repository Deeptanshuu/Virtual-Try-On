import os
import requests
import logging
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from preprocess.humanparsing.run_parsing import Parsing
from src.image_format_convertor import ImageFormatConvertor

REMOVE_BG_KEY = os.getenv('REMOVE_BG_KEY')

parsing_model = Parsing(0)

class BackgroundProcessor:
    DeprecationWarning("Created only for testing. Not in use")
    @classmethod
    def add_background(cls, human_img: Image, background_img: Image):
        
        human_img = human_img.convert("RGB")
        width = human_img.width
        height = human_img.height
        
        # Create mask image
        parsed_img, _ = parsing_model(human_img)
        mask_img = parsed_img.convert("L")
        mask_img = mask_img.resize((width, height))
        
        background_img = background_img.convert("RGB")
        background_img = background_img.resize((width, height))

        # Convert to numpy arrays
        human_np = np.array(human_img)
        mask_np = np.array(mask_img)
        background_np = np.array(background_img)

        # Ensure mask is 3-channel (RGB) for compatibility
        mask_np = np.stack((mask_np,) * 3, axis=-1)

        # Apply the mask to human_img
        human_with_background = np.where(mask_np > 0, human_np, background_np)

        # Convert back to PIL Image
        result_img = Image.fromarray(human_with_background.astype('uint8'))

        # Return or save the result
        return result_img

    DeprecationWarning("Created only for testing. Not in use")
    @classmethod
    def add_background_v3(cls, foreground_pil: Image, background_pil: Image):
        foreground_pil= foreground_pil.convert("RGB")
        width = foreground_pil.width
        height = foreground_pil.height

        # Create mask image
        parsed_img, _ = parsing_model(foreground_pil)
        mask_pil = parsed_img.convert("L")
        # Apply a threshold to convert to binary image
        # mask_pil = mask_pil.point(lambda p: 1 if p > 127 else 0, mode='1')
        mask_pil = mask_pil.resize((width, height))
        
        # Resize background image
        background_pil = background_pil.convert("RGB")
        background_pil = background_pil.resize((width, height))
        
        # Load the images using PIL
        #foreground_pil = Image.open(human_img_path).convert("RGB")  # The segmented person image
        #background_pil = Image.open(background_img_path).convert("RGB")  # The new background image
        #mask_pil = Image.open(mask_img_path).convert('L')  # The mask image from the human parser model

        # Resize the background to match the size of the foreground
        #background_pil = background_pil.resize(foreground_pil.size)

        # Resize mask
        #mask_pil = mask_pil.resize(foreground_pil.size)

        # Convert PIL images to OpenCV format
        foreground_cv2 = ImageFormatConvertor.pil_to_cv2(foreground_pil)
        background_cv2 = ImageFormatConvertor.pil_to_cv2(background_pil)
        #mask_cv2 = pil_to_cv2(mask_pil)
        mask_cv2 = np.array(mask_pil)  # Directly convert to NumPy array without color conversion

        # Ensure the mask is a single channel image
        if len(mask_cv2.shape) == 3:
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)

        # Threshold the mask to convert it to pure black and white
        _, mask_cv2 = cv2.threshold(mask_cv2, 0, 255, cv2.THRESH_BINARY)

        # Ensure the mask is a single channel image
        #if len(mask_cv2.shape) == 3:
        #    mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)

        # Create an inverted mask
        mask_inv_cv2 = cv2.bitwise_not(mask_cv2)

        # Convert mask to 3 channels
        mask_3ch_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_GRAY2BGR)
        mask_inv_3ch_cv2 = cv2.cvtColor(mask_inv_cv2, cv2.COLOR_GRAY2BGR)

        # Extract the person from the foreground image using the mask
        person_cv2 = cv2.bitwise_and(foreground_cv2, mask_3ch_cv2)

        # Extract the background where the person is not present
        background_extracted_cv2 = cv2.bitwise_and(background_cv2, mask_inv_3ch_cv2)

        # Combine the person and the new background
        combined_cv2 = cv2.add(person_cv2, background_extracted_cv2)

        # Refine edges using Gaussian Blur (feathering technique)
        blurred_combined_cv2 = cv2.GaussianBlur(combined_cv2, (5, 5), 0)

        # Convert the result back to PIL format
        combined_pil = ImageFormatConvertor.cv2_to_pil(blurred_combined_cv2)
        

        """
        # Post-processing: Adjust brightness, contrast, etc. (optional)
        enhancer = ImageEnhance.Contrast(combined_pil)
        post_processed_pil = enhancer.enhance(1.2)  # Adjust contrast
        enhancer = ImageEnhance.Brightness(post_processed_pil)
        post_processed_pil = enhancer.enhance(1.2)  # Adjust brightness
        """


        # Save the final image
        # post_processed_pil.save('path_to_save_final_image_1.png')

        # Display the images (optional)
        #foreground_pil.show(title="Foreground")
        #background_pil.show(title="Background")
        #mask_pil.show(title="Mask")
        #combined_pil.show(title="Combined")
        # post_processed_pil.show(title="Post Processed")

        return combined_pil
    
    DeprecationWarning("Created only for testing. Not in use")
    @classmethod
    def replace_background(cls, foreground_img_path: str, background_img_path: str):
        # Load the input image (with alpha channel) and the background image
        #input_image = cv2.imread(foreground_img_path, cv2.IMREAD_UNCHANGED)        
        # background_image = cv2.imread(background_img_path)
        foreground_img_pil = Image.open(foreground_img_path)
        width = foreground_img_pil.width
        height = foreground_img_pil.height
        background_image_pil = Image.open(background_img_path)
        background_image_pil = background_image_pil.resize((width, height))
        input_image = ImageFormatConvertor.pil_to_cv2(foreground_img_pil)
        background_image = ImageFormatConvertor.pil_to_cv2(background_image_pil)
        

        # Ensure the input image has an alpha channel
        if input_image.shape[2] != 4:
            raise ValueError("Input image must have an alpha channel")

        # Extract the alpha channel
        alpha_channel = input_image[:, :, 3]

        # Resize the background image to match the input image dimensions
        background_image = cv2.resize(background_image, (input_image.shape[1], input_image.shape[0]))

        # Convert alpha channel to 3 channels
        alpha_channel_3ch = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
        alpha_channel_3ch = alpha_channel_3ch / 255.0  # Normalize to 0-1

        # Extract the BGR channels of the input image
        input_bgr = input_image[:, :, :3]
        background_bgr = background_image[:,:,:3]
        # Blend the images using the alpha channel
        foreground = cv2.multiply(alpha_channel_3ch, input_bgr.astype(float))
        background = cv2.multiply(1.0 - alpha_channel_3ch, background_bgr.astype(float))
        combined_image = cv2.add(foreground, background).astype(np.uint8)

        # Save and display the result
        cv2.imwrite('path_to_save_combined_image.png', combined_image)
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    
    @classmethod
    def replace_background_with_removebg(cls, foreground_img_pil: Image, background_image_pil: Image):
        foreground_img_pil= foreground_img_pil.convert("RGB")
        width = foreground_img_pil.width
        height = foreground_img_pil.height

        # Resize background image
        background_image_pil = background_image_pil.convert("RGB")
        background_image_pil = background_image_pil.resize((width, height))

        #foreground_img_pil = Image.open(foreground_img_path)
        #width = foreground_img_pil.width
        #height = foreground_img_pil.height
        #background_image_pil = Image.open(background_img_path)
        #background_image_pil = background_image_pil.resize((width, height)) 
        
        # Do color transfer of background to foreground to adjust lighting condition
        #foreground_img_pil = cls.color_transfer(foreground_img_pil, background_image_pil)

        foreground_binary = ImageFormatConvertor.pil_image_to_binary_data(foreground_img_pil)
        background_binary = ImageFormatConvertor.pil_image_to_binary_data(background_image_pil)
        combined_img_pil = cls.remove_bg(foreground_binary, background_binary)        
        return combined_img_pil


    @classmethod
    def remove_bg(cls, foreground_binary: str, background_binary: str):
        # ref: https://www.remove.bg/api#api-reference
        url = "https://api.remove.bg/v1.0/removebg"        
        
        # using form-data as passing binary data is not supported in application/json
        files = {
            "image_file": ('foreground.png', foreground_binary, 'image/png'),
            "bg_image_file": ('background.png', background_binary,  'image/png')
        }

        # get output image in same resolution as input
        payload = {
            "size": "full",
            "shadow_type": "3D"
        }        
        headers = {
            "accept": "image/*",
            'X-Api-Key': REMOVE_BG_KEY
        } 
        remove_bg_request = requests.post(url, files=files, data=payload, headers=headers, timeout=20)
        if remove_bg_request.status_code == 200:
            image_content = remove_bg_request.content
            pil_image = ImageFormatConvertor.binary_data_to_pil_image(image_content)
            return pil_image
        logging.error(f"failed to use remove bg. Status: {remove_bg_request.status_code}. Resp: {remove_bg_request.content}")
        return None
    
    @classmethod
    def create_mask(cls, foreground_path: str, mask_path: str):
        """
        Given foreground image path with background removed, create a maska and save it in mask_path
        """
        # Load the foreground image with alpha channel
        foreground = Image.open(foreground_path)

        # Convert to RGBA if not already
        foreground = foreground.convert("RGBA")

        # Create the mask from the alpha channel
        alpha_channel = np.array(foreground.split()[-1])

        # Create a binary mask where alpha > 0 is white (255) and alpha == 0 is black (0)
        mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)

        # Save the mask to a file
        Image.fromarray(mask).save(mask_path) 
    
    @classmethod
    def get_minimal_bounding_box(cls, foreground_pil: Image):
        """
        Result x1,y1,x2,y2 ie cordinate of bottom left and top right
        """
        # convert to cv2
        foreground = ImageFormatConvertor.pil_to_cv2(foreground_pil)
        # Ensure the image has an alpha channel (transparency)
        if foreground.shape[2] == 4:
            # Extract the alpha channel
            alpha_channel = foreground[:, :, 3]            
            # Create a binary image from the alpha channel
            _, binary_image = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        else:
            # If there is no alpha channel, convert the image to grayscale
            gray_image = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)            
            # Apply binary thresholding
            _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        
        # Find all non-zero points (non-background)
        non_zero_points = cv2.findNonZero(binary_image)
        
        # Get the minimal bounding rectangle
        if non_zero_points is not None:
            x, y, w, h = cv2.boundingRect(non_zero_points)
            """
            # Optionally, draw the bounding box on the image for visualization
            output_image = foreground.copy()
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0, 255), 2)
            # Save or display the output image
            output_image_pil = ImageFormatConvertor.cv2_to_pil(output_image)
            output_image_pil.save('output_with_bounding_box.png')
            """
            
            return (x, y, x + w, y + h)
        else:
            return 0,0,w,h
    
    @classmethod
    def color_transfer(cls, source_pil: Image, target_pil: Image) -> Image:
        # NOT IN USE as output color was not good
        source = ImageFormatConvertor.pil_to_cv2(source_pil)
        # Resize background image
        width, height = source_pil.width, source_pil.height
        target_pil = target_pil.convert("RGB")
        target_pil = target_pil.resize((width, height))

        target = ImageFormatConvertor.pil_to_cv2(target_pil)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        # Compute the mean and standard deviation of the source and target images
        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)

        #Reshape the mean and std to (1, 1, 3) so they can be broadcast correctly
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))
        # Subtract the mean from the source image
        result = (source - source_mean) * (target_std / source_std) + target_mean
        result = np.clip(result, 0, 255).astype(np.uint8)

        res = cv2.cvtColor(result, cv2.COLOR_LAB2BGR) 
        res_pil = ImageFormatConvertor.cv2_to_pil(res)
        return res_pil

    @classmethod
    def intensity_transfer(cls, source_pil: Image, target_pil: Image) -> Image:

        """
        Transfers the intensity distribution from the target image to the source image.

        Parameters:
        source (np.ndarray): The source image (foreground) to be harmonized.
        target (np.ndarray): The target image (background) whose intensity distribution is to be matched.
        eps (float): A small value to avoid division by zero.

        Returns:
        np.ndarray: The intensity-transferred source image.
        """
        source = ImageFormatConvertor.pil_to_cv2(source_pil)
        # Resize background image
        width, height = source_pil.width, source_pil.height
        target_pil = target_pil.convert("RGB")
        target_pil = target_pil.resize((width, height))

        target = ImageFormatConvertor.pil_to_cv2(target_pil)

        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        # Compute the mean and standard deviation of the L channel (intensity) of the source and target images
        source_mean, source_std = cv2.meanStdDev(source_lab[:, :, 0])
        target_mean, target_std = cv2.meanStdDev(target_lab[:, :, 0])

        # Reshape the mean and std to (1, 1, 1) so they can be broadcast correctly
        source_mean = source_mean.reshape((1, 1, 1))
        source_std = source_std.reshape((1, 1, 1))
        target_mean = target_mean.reshape((1, 1, 1))
        target_std = target_std.reshape((1, 1, 1))

        # Transfer the intensity (L channel)
        result_l = (source_lab[:, :, 0] - source_mean) * (target_std / source_std) + target_mean
        result_l = np.clip(result_l, 0, 255).astype(np.uint8)

        # Combine the transferred L channel with the original A and B channels
        result_lab = np.copy(source_lab)
        result_lab[:, :, 0] = result_l

        # return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        res = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR) 
        res_pil = ImageFormatConvertor.cv2_to_pil(res)
        return res_pil
    
    @classmethod
    def match_color(cls, source_pil: Image, target_pil: Image):
        source = ImageFormatConvertor.pil_to_cv2(source_pil)
        # Resize background image
        width, height = source_pil.width, source_pil.height
        target_pil = target_pil.convert("RGB")
        target_pil = target_pil.resize((width, height))

        target = ImageFormatConvertor.pil_to_cv2(target_pil)
        
        matched_foreground = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        matched_background = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # Match the histograms
        for i in range(3):
            matched_foreground[:, :, i] = cv2.equalizeHist(matched_foreground[:, :, i])
            matched_background[:, :, i] = cv2.equalizeHist(matched_background[:, :, i])
        
        matched_foreground = cv2.cvtColor(matched_foreground, cv2.COLOR_LAB2BGR)
        matched_background = cv2.cvtColor(matched_background, cv2.COLOR_LAB2BGR)

        matched_foreground_pil = ImageFormatConvertor.cv2_to_pil(matched_foreground)
        matched_background_pil = ImageFormatConvertor.cv2_to_pil(matched_background)

        return matched_foreground_pil, matched_background_pil

