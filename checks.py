import datetime
import os
import re
from PIL import Image

import boto3
import cv2
import imutils
import jellyfish
import pytesseract
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
from pytesseract import TesseractError


class AWSService:
    s3_client = boto3.client(
        's3', region_name='us-east-2',
        aws_access_key_id='AKIAWYRV6LCIHYS27NWN',
        aws_secret_access_key='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'
    )
    textract_client = boto3.client(
        'textract', region_name='us-east-2',
        aws_access_key_id='AKIAWYRV6LCIHYS27NWN',
        aws_secret_access_key='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'
    )

    @classmethod
    def process_file_with_textract(cls, file_path: str) -> dict:
        """
        Method making request to AWS Textract and returning response - info
        about text in image
        :param file_path: file path relative root project directory
        :return: dictionary with image text info
        """

        with open(file_path, 'rb') as image:
            img = bytearray(image.read())
            response = cls.textract_client.detect_document_text(
                Document={'Bytes': img}
            )
            return response

    @classmethod
    def get_document_bounding_box_coordinates(cls, file_path: str) -> list:
        """
        Method gets relative file path to root project directory,
        returns list of top left corner coordinates and width, height length of
        document on image
        :param file_path: file path relative root project directory
        :return: list of top, left coordinates and width, height length
        """

        response = cls.process_file_with_textract(file_path)
        blocks_list = response['Blocks']
        page = blocks_list[0]
        b_box = page['Geometry']['BoundingBox']
        top = b_box['Top']
        left = b_box['Left']
        height = b_box['Height']
        width = b_box['Width']
        bounding_box = [top, left, height, width]
        return bounding_box

    @classmethod
    def get_text_from_local_image(cls, file_name: str) -> str:
        """
        Method gets image file path in param and reads it with AWS Textract,
        then gets all text from image and returns all text in one string
        :param file_name: image file path
        :return: string with text from image
        """

        response = cls.process_file_with_textract(file_name)
        page_text = ''
        for item in response['Blocks']:
            if 'Text' in item.keys():
                page_text += (item['Text'].lower()) + ' '

        return page_text


class FilesService:
    @staticmethod
    def convert_pdf_to_jpeg_from_bytes(file: bytes) -> list:
        """
        Method gets pdf file in bytes type as argument and returns list of
        images converted from pdf bytes file
        :param file: bytes pdf file
        :return: list of images
        """

        images = convert_from_bytes(
            file.getvalue(),
        )
        return images

    @staticmethod
    def convert_pdf_to_jpeg_from_path(file_path: str) -> list:
        """
        Method gets pdf file by path as argument and returns list of
        images converted from pdf file
        :param file_path: pdf file path
        :return: list of images
        """

        images = convert_from_path(
            os.path.abspath(file_path),
        )
        return images

    @classmethod
    def save_temp_images(cls, images: list) -> list:
        """
        Method gets list of images by param, generating temporary file names
        and returns list of temporary images path
        :param images: list of images
        :return: list of temp images path
        """

        file_names = []
        for image in images:
            file_name = cls.generate_temp_file_name()
            try:
                image.save(file_name, 'JPEG')
                file_names.append(file_name)
            except:
                image = Image.open(image)
                image.save(file_name, 'JPEG')
                file_names.append(file_name)

        return file_names

    @staticmethod
    def generate_temp_file_name() -> str:
        """
        Method generates file name and create directory if it is not exist
        :return: temp file name
        """

        file_path = 'services/temp/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_name = file_path + str(
            datetime.datetime.now()
        ).replace(' ', '_') + '.jpg'
        return file_name

    @staticmethod
    def generate_cropped_file_name() -> str:
        """
        Method generates file name to save cropped image and creating directory
        if it does not exist, then returns cropped file name
        :return: cropped file name
        """

        file_path = 'services/cropped/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_name = file_path + str(
            datetime.datetime.now()
        ).replace(' ', '_') + '.jpg'
        return file_name

    @staticmethod
    def generate_rotated_file_name() -> str:
        """
        Method generates file name to save rotated image and creating directory
        if it does not exist, then returns rotated file name
        :return: rotated file name
        """

        file_path = 'services/rotated/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_name = file_path + str(
            datetime.datetime.now()
        ).replace(' ', '_') + '.jpg'
        return file_name


class CheckCropService:
    @classmethod
    def crop_with_opencv(cls, file_name: str) -> list:
        """
        Method gets file path in param file_name, read with opencv and converts
        it to format easy to crop, then crop it and returns list with cropped
        file path and cropped image
        :param file_name: file path to crop
        :return: list with cropped file path and cropped image
        """

        cropped_file = FilesService.generate_cropped_file_name()
        image = cv2.imread(file_name)

        # convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((7, 7), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get largest contour
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        area_thresh = 0
        big_contour = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_thresh:
                area_thresh = area
                big_contour = contour

        # get bounding box
        x, y, w, h = cv2.boundingRect(big_contour)
        rect_image = cv2.rectangle(
            image, (x, y), (x + w, y + h), (0, 255, 0), 2
        )

        # apply mask to input
        result = image.copy()
        result = cv2.bitwise_and(result, rect_image)

        # crop result
        cropped = result[y:y + h, x:x + w]
        cv2.imwrite(cropped_file, cropped)
        cropped_list = [cropped_file, cropped]
        return cropped_list

    @classmethod
    def crop_by_textract_percentage(cls, file_name: str) -> list:
        """
        Method gets file path in param file_name, read with AWS Textract
        and read it with opencv to crop, then crop it by percentage and returns
        list with cropped file path and cropped image
        :param file_name: file path
        :return: list with cropped image path and cropped image
        """

        cropped_file = FilesService.generate_cropped_file_name()
        left, top, width, height = (
            AWSService
            .get_document_bounding_box_coordinates(file_name)
        )
        image = cv2.imread(file_name)

        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        width_scaled, height_scaled = (
            image.shape[1] * width, image.shape[0] * height
        )
        left_x, right_x = (
            center_x - width_scaled / 2, center_x + width_scaled / 2
        )
        top_y, bottom_y = (
            center_y - height_scaled / 2, center_y + height_scaled / 2
        )

        cropped = image[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        cv2.imwrite(cropped_file, cropped)
        cropped_list = [cropped_file, cropped]
        return cropped_list

    @classmethod
    def crop_images(cls, file_names: list) -> dict:
        """
        Method runs cropping process for files in params,
        returns dict with status
        :param file_names: list of files paths
        :return: dict with crop status, and path to cropped images
        """
        
        cropped_files = []
        for file_name in file_names:
            try:
                # cropped, image = cls.crop_by_textract_percentage(file_name)
                cropped, image = cls.crop_with_opencv(file_name)
                # cls.validate_correct_crop(file_name, cropped, image)
                cropped_files.append(cropped)
            except:
                return {"status": "Some error occurred"}
        return {"status": "Cropped successfully", "files": cropped_files}


class CheckRotateService:
    @classmethod
    def imshow(cls, name, image) -> None:
        """
        Debug method to see image pre-rotation and after-rotation
        :param name: window name
        :param image: image read with opencv
        :return: None
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1000, 500)
        cv2.imshow(name, image)

    @classmethod
    def rotate_image(cls, file_name):
        """
        Method gets image path in param file_name and returns dict with status
        and rotated file path
        :param file_name: image path
        :return: dict with status and rotated file path
        """
        
        rotated_file = FilesService.generate_rotated_file_name()
        image = cv2.imread(file_name)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        gray = cv2.medianBlur(gray, 3)

        ret, threshold = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )

        modified = threshold
        modified = cv2.medianBlur(modified, 3)
        modified = cv2.Canny(modified, 100, 200)

        try:
            rotation_data = pytesseract.image_to_osd(
                modified, config='-c min_characters_to_try=40'
            )
            rotation_angle = re.search(
                '(?<=Rotate: )\d+', rotation_data
            ).group(0)
            angle = float(rotation_angle)
            rotated = imutils.rotate_bound(image, angle)
            cv2.imwrite(rotated_file, rotated)
            return {
                "status": "Image rotated successfully.",
                "file_path": rotated_file
            }
        except (TesseractError, cv2.error):
            return {"status": "Rotation failed."}


class CheckValidationService:

    @classmethod
    def get_text_for_validation(cls, file_path: str) -> list:
        """
        Method sends request to AWS and gets response with data from image,
        creates list with lines of text from image and returns it to compare
        after crop
        :param file_path: string path of image file
        :return: list of text lines
        """

        response = AWSService.process_file_with_textract(file_path)
        blocks_list = response['Blocks']
        lines_list = []
        for block in blocks_list:
            if block['BlockType'] == 'LINE':
                lines_list.append(block['Text'])

        return lines_list

    @classmethod
    def detect_page(cls, page_text: str) -> dict:
        """
        Method gets page text in param and compare it with key words to find
        front page and back page
        :param page_text: string with text from image
        :return: dict with page status
        """

        front_page_words = [
            'claim', 'policy', 'loss', 'insured', 'insurance', 'company'
        ]
        matched_keywords = 0
        for word in front_page_words:
            word = page_text.find(word)
            if word != -1:
                matched_keywords += 1

        if matched_keywords >= 3:
            return {"page_status": "Front page"}
        else:
            return {"page_status": "Back page"}

    @classmethod
    def validate_correct_crop(cls, file_name: str, cropped: str) -> dict:
        """
        Method validating correct crop, without lost of information, it compare
        image text before and after and returns dict with status
        :param file_name: string file path before crop
        :param cropped: string file path after crop
        :return: dict with status
        """

        before_lines = cls.get_text_for_validation(file_name)
        after_lines = cls.get_text_for_validation(cropped)
        first_line_validation = []
        last_line_validation = []
        for after_line_start, after_line_end in zip(after_lines[0:6],
                                                    after_lines[-1:-6:-1]):
            first_line_distance = jellyfish.levenshtein_distance(
                before_lines[0], after_line_start
            )
            last_line_distance = jellyfish.levenshtein_distance(
                before_lines[-1], after_line_end
            )
            first_line_validation.append(first_line_distance)
            last_line_validation.append(last_line_distance)
        if min(first_line_validation) < 10 and min(last_line_validation) < 10:
            return {"status": "Valid crop."}
        else:
            return {"status": "Invalid crop."}


class ProcessService:
    @staticmethod
    def pdf_processing(file):
        images = FilesService.convert_pdf_to_jpeg_from_bytes(file)
        file_names = FilesService.save_temp_images(images)
        response = CheckCropService.crop_images(file_names)
        files = response['files']
        for file in files:
            rotate_response = CheckRotateService.rotate_image(file)
            if rotate_response['status'] == 'Rotation failed.':
                return False
        return True

    @staticmethod
    def image_processing(images):
        file_names = FilesService.save_temp_images(images)
        response = CheckCropService.crop_images(file_names)
        files = response['files']
        for file in files:
            rotate_response = CheckRotateService.rotate_image(file)
            if rotate_response['status'] == 'Rotation failed.':
                return False
        return True
