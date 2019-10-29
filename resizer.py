from PIL import Image

im = Image.open("not_resized/white/Image.jpg")
im2 = im.resize((224, 224), Image.BICUBIC)
# im.show()
print(im)
im2.save("test_creature_color/white/image.png", "png")