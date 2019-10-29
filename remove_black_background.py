import cv2
from PIL import Image, ImageChops

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

im = Image.open("test_creature_color/green/test_2_nobg.jpg")
im = trim(im)
im.show()
print(im)
im.save("test_creature_color/green/test_2_noblackbg.jpg", "png")