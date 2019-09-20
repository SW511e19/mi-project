try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename), lang='eng')  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

#pure magic
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\samuel\AppData\Local\Tesseract-OCR\tesseract.exe'
image_name = 'demon'
print(ocr_core(image_name + '.png'))


