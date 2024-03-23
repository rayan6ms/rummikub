import cv2
import numpy as np
import os
import shutil
from mss import mss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

snips_folder = "snips"
training_folder = "knn"
output_folder = "output"
extracted_numbers = "extracted_numbers"
table_image = "tiles.png"
pfp_image = "pfp.png"


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def capture_screen(region):
    with mss() as sct:
        sct_img = sct.grab(region)
        return np.array(sct_img)[:, :, :3]


board_region = {"top": 280, "left": 210, "width": 835, "height": 440}
board_screenshot = capture_screen(board_region)

screen = capture_screen(board_region)
cv2.imwrite("tiles.png", screen)
print("Screenshot salva como tiles.png.")


def apply_color_filter_to_tiles(image_path):
    img = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros(hsv_image.shape[:2], dtype="uint8")
    pink_hsv = np.array([160, 255, 255])

    for snip_filename in os.listdir(snips_folder):
        snip_path = os.path.join(snips_folder, snip_filename)
        snip_img = cv2.imread(snip_path)
        snip_hsv = cv2.cvtColor(snip_img, cv2.COLOR_BGR2HSV)
        average_color = np.mean(snip_hsv, axis=(0, 1))

        lower_bound = average_color * 0.9
        upper_bound = average_color * 1.1

        mask = cv2.inRange(
            hsv_image, lower_bound.astype("uint8"), upper_bound.astype("uint8")
        )
        mask_total = cv2.bitwise_or(mask_total, mask)

    hsv_image[mask_total > 0] = pink_hsv
    img_processed = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return img_processed


def combine_close_contours(contours, max_distance=5, padding=3):
    bounding_boxes = [
        apply_padding(
            cv2.boundingRect(contour),
            padding
        ) for contour in contours
    ]
    combined_boxes = []

    while bounding_boxes:
        base = bounding_boxes.pop(0)
        merged = True
        while merged:
            merged = False
            for i, box in enumerate(bounding_boxes):
                if is_close(base, box, max_distance):
                    base = merge_boxes(base, box)
                    bounding_boxes.pop(i)
                    merged = True
                    break
        combined_boxes.append(base)

    expanded_contours = [box_to_contour(box) for box in combined_boxes]
    return expanded_contours


def apply_padding(box, padding):
    x, y, w, h = box
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding
    return (x, y, w, h)


def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y

    return (x, y, w, h)


def box_to_contour(box):
    x, y, w, h = box
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])


def is_close(box1, box2, max_distance):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    distance_x = max(x1, x2) - min(x1 + w1, x2 + w2)
    distance_y = max(y1, y2) - min(y1 + h1, y2 + h2)

    return max(distance_x, distance_y) <= max_distance


def cut_and_pad(image, size=(48, 48)):
    h, w = image.shape[:2]
    if h > size[1] or w > size[0]:
        excess_height = max(0, h - size[1])
        excess_width = max(0, w - size[0])
        image = image[
            excess_height // 2: h - (excess_height // 2 + excess_height % 2),
            excess_width // 2: w - (excess_width // 2 + excess_width % 2),
        ]
    new_h, new_w = image.shape[:2]
    if new_h < size[1] or new_w < size[0]:
        padding_top = max(0, (size[1] - new_h) // 2)
        padding_bottom = max(0, size[1] - new_h - padding_top)
        padding_left = max(0, (size[0] - new_w) // 2)
        padding_right = max(0, size[0] - new_w - padding_left)
        image = cv2.copyMakeBorder(
            image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    return image


def sort_contours(contours):
    centered_contours = [
        (
            c,
            cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2
        ) for c in contours
    ]

    lines = {}
    for contour, center_y in centered_contours:
        found_line = False
        for line in lines:
            if abs(center_y - line) < 10:
                lines[line].append(contour)
                found_line = True
                break
        if not found_line:
            lines[center_y] = [contour]

    sorted_lines_keys = sorted(lines.keys())

    sorted_contours = []
    for line_key in sorted_lines_keys:
        line_contours = sorted(
            lines[line_key], key=lambda c: cv2.boundingRect(c)[0])
        sorted_contours.extend(line_contours)

    return sorted_contours


def process_and_extract_numbers(image_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(extracted_numbers):
        os.makedirs(extracted_numbers)

    original_img = cv2.imread(image_path)
    img_processed = apply_color_filter_to_tiles(image_path)

    hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)

    colors = {
        "B": ((100, 150, 0), (140, 255, 255)),
        "R": ((0, 70, 50), (10, 255, 255)),
        "R2": ((170, 70, 50), (180, 255, 255)),
        "K": ((0, 0, 0), (180, 255, 50)),
        "Y": ((10, 100, 100), (25, 255, 255)),
    }

    mask_total = np.zeros(hsv.shape[:2], dtype="uint8")
    color_masks = {}
    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_total = cv2.bitwise_or(mask_total, mask)
        color_masks[color] = mask

    img_processed[mask_total > 0] = [160, 255, 255]
    img_processed[mask_total == 0] = [0, 0, 0]

    contours, _ = cv2.findContours(
        mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    combined_contours = combine_close_contours(contours)
    sorted_contours = sort_contours(combined_contours)

    cv2.drawContours(img_processed, sorted_contours, -1, (0, 255, 0), 2)

    clear_directory(extracted_numbers)

    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        color_name = "Unknown"
        max_intersection = 0
        for color, mask in color_masks.items():
            intersection = np.sum(mask[y: y + h, x: x + w])
            if intersection > max_intersection:
                max_intersection = intersection
                color_name = color

        number_img = original_img[y: y + h, x: x + w]
        gray_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        resized_img = cut_and_pad(binary_img)

        cv2.imwrite(
            os.path.join(extracted_numbers,
                         f"{i}_{color_name}.png"), resized_img
        )

    contours_image_path = os.path.join(output_folder, "contours.png")
    cv2.imwrite(contours_image_path, img_processed)


def prepare_images_for_knn():
    X = []
    y = []

    for img_name in os.listdir(training_folder):
        if img_name.endswith(".png"):
            img_path = os.path.join(training_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            flattened = img.flatten()
            X.append(flattened)

            label = int(img_name.split()[0].split("(")[0])
            y.append(label)

    return np.array(X), np.array(y)


def predict_images():
    X_processed = []
    file_names = []

    for img_name in sorted(
        os.listdir(extracted_numbers), key=lambda x: int(x.split("_")[0])
    ):
        if img_name.endswith(".png"):
            img_path = os.path.join(extracted_numbers, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            flattened = img.flatten()
            X_processed.append(flattened)
            file_names.append(img_name)

    X_processed = np.array(X_processed)
    predictions = knn.predict(X_processed)
    return dict(zip(file_names, predictions))


process_and_extract_numbers(table_image)

X, y = prepare_images_for_knn()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

image_predictions = predict_images()
predictions = []

for img_name, prediction in image_predictions.items():
    name_without_number = img_name.split("_")[1].split(".")[0]
    if "R2" in img_name:
        prediction = 0
    predictions.append(f"{prediction}_{name_without_number}")

print("KNN:", predictions)
