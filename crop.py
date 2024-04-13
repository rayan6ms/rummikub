from PIL import Image


def crop_image(input_path, output_path, top, bottom, left, right):
    with Image.open(input_path) as img:
        width, height = img.size
        left = width * (left / 100)
        right = width - (width * (right / 100))
        top = height * (top / 100)
        bottom = height - (height * (bottom / 100))

        crop_area = (left, top, right, bottom)
        cropped_img = img.crop(crop_area)
        cropped_img.save(output_path)


input_image = "game.png"

crop_image(input_image, "test/table.png", 5, 28, 13, 12)
crop_image(input_image, "test/rack.png", 76.4, 1, 22.6, 22.9)
