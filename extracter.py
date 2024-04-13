import cv2
import numpy as np
import os
import shutil
import asyncio
import base64
from PIL import Image
import io
import websockets
import json
from mss import mss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from set_generator import SetGenerator
from solver import RummikubSolver
from collections import Counter

tiles_path = 'tiles'
snips_folder = "snips"
training_folder = "knn"
output_folder = "output"
extracted_numbers = "extracted_numbers"
table_image = "table.png"
rack_image = "rack.png"
game_image = "game.png"


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def capture_screen(region):
    with mss() as sct:
        sct_img = sct.grab(region)
        return np.array(sct_img)[:, :, :3]


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


def group_contours(sorted_contours, group_distance=44):
    groups = []
    current_group = []
    last_x_end = 0
    last_y_center = None

    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_center = y + h // 2

        if (last_y_center is not None and abs(y_center - last_y_center) > h // 2) or (current_group and x - last_x_end > group_distance):
            groups.append(current_group)
            current_group = []

        current_group.append(contour)
        last_x_end = x + w
        last_y_center = y_center

    if current_group:
        groups.append(current_group)

    return groups


def process_and_extract_numbers(image_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(extracted_numbers):
        os.makedirs(extracted_numbers)

    original_img = cv2.imread(image_path)

    img_processed = apply_color_filter_to_tiles(image_path)
    cv2.imwrite(os.path.join(output_folder, "color_filter.png"), img_processed)

    hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)

    colors = {
        "B": ((100, 150, 0), (140, 255, 255)),
        "R": ((0, 150, 150), (10, 255, 255)),
        "R2": ((170, 100, 100), (180, 255, 255)),
        "K": ((0, 0, 0), (180, 255, 50)),
        "Y": ((10, 100, 100), (25, 255, 255)),
    }

    mask_total = np.zeros(hsv.shape[:2], dtype="uint8")
    color_masks = {}
    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_total = cv2.bitwise_or(mask_total, mask)
        color_masks[color] = mask
        cv2.imwrite(os.path.join(output_folder, f"mask_{color}.png"), mask)

    img_processed[mask_total > 0] = [160, 255, 255]
    img_processed[mask_total == 0] = [0, 0, 0]
    cv2.imwrite(os.path.join(output_folder, "mask_applied.png"), img_processed)

    contours, _ = cv2.findContours(
        mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    combined_contours = combine_close_contours(contours)
    sorted_contours = sort_contours(combined_contours)
    groups = group_contours(sorted_contours)

    cv2.drawContours(img_processed, sorted_contours, -1, (0, 255, 0), 2)

    clear_directory(extracted_numbers)

    group_number = 0
    index = 0
    for group in groups:
        for contour in group:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 or h > 10:
                color_name = "Unknown"
                max_intersection = 0
                for color, mask in color_masks.items():
                    intersection = np.sum(mask[y: y + h, x: x + w])
                    if intersection > max_intersection:
                        max_intersection = intersection
                        color_name = color

                number_img = original_img[y: y + h, x: x + w]
                cv2.imwrite(os.path.join('output/extracted',
                            f"{index}_extracted.png"), number_img)
                gray_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
                _, binary_img = cv2.threshold(
                    gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                resized_img = cut_and_pad(binary_img)

                cv2.imwrite(os.path.join(
                    extracted_numbers,
                    f"{index}_{color_name}_{group_number}.png"
                ), resized_img)
                index += 1
        group_number += 1

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


def predict_images(knn):
    X_processed = []
    file_names = []

    groups = {}

    for img_name in sorted(
        os.listdir(extracted_numbers),
        key=lambda x: (
            int(x.split("_")[-1].split(".")[0]),
            int(x.split("_")[0])
        )
    ):
        if img_name.endswith(".png"):
            group_number = img_name.rsplit("_", 1)[-1].split(".")[0]
            img_path = os.path.join(extracted_numbers, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            flattened = img.flatten()
            X_processed.append(flattened)
            file_names.append(img_name)
            if group_number not in groups:
                groups[group_number] = []

    X_processed = np.array(X_processed)
    predictions = knn.predict(X_processed)

    for img_name, prediction in zip(file_names, predictions):
        group_number = img_name.rsplit("_", 1)[-1].split(".")[0]
        name_without_ext = img_name.rsplit(".", 1)[0]
        name_without_number = "_".join(name_without_ext.split("_")[1:-1])
        if "R2" in img_name:
            prediction = 0
            name_without_number = name_without_number[:-1]
        groups[group_number].append(f"{prediction}_{name_without_number}")

    grouped_predictions = [groups[group]
                           for group in sorted(groups, key=lambda x: int(x))]
    return grouped_predictions


def get_tiles(image_path):
    process_and_extract_numbers(image_path)

    X, y = prepare_images_for_knn()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    return predict_images(knn)


def get_solution(solver, r_tile_map, rack, maximise='tiles'):
    value, tiles, sets = solver.solve(maximise=maximise)
    if value == 0:
        image_path = 'game/rummikub.png'
        img = cv2.imread(image_path)

        blurred_img = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)

        brightness_reducer = np.ones(img.shape, dtype="uint8") * 50
        dim_img = cv2.subtract(blurred_img, brightness_reducer)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Buy new tile"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textX = (dim_img.shape[1] - textsize[0]) / 2
        textY = (dim_img.shape[0] / 2) - 20
        cv2.putText(dim_img, text, (int(textX), int(textY)),
                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(image_path, dim_img)

        return None, None, None, True
    else:
        tile_list = [solver.tiles[i]
                     for i in range(len(tiles)) if tiles[i] == 1]
        highlighted_tiles = [r_tile_map[t] for t in tile_list]

        set_list = [solver.sets[i] for i in range(len(sets)) if sets[i] == 1]
        new_table = [[r_tile_map[t] for t in s] for s in set_list]

        highlighted_count = Counter(highlighted_tiles)
        rack_count = Counter(rack)

        remaining_rack_count = rack_count - highlighted_count

        remaining_rack = list(remaining_rack_count.elements())

        return highlighted_tiles, new_table, remaining_rack, False


def create_number_maps(sg):
    colours = ['K', 'B', 'Y', 'R']
    verbose_list = [f'{n}_{colours[c]}' for c in range(
        sg.colours) for n in range(1, sg.numbers + 1)]

    verbose_list.append('j')

    tile_map = dict(zip(verbose_list, sg.tiles +
                        [sg.tiles[-1] + 1]))
    r_tile_map = {v: k for k, v in tile_map.items()}

    return tile_map, r_tile_map


def replace_jokers(tile):
    return 'j' if tile in ['0_R', '0_K'] else tile


def revert_joker(tile, original_table, original_rack):
    if tile == 'j':
        if '0_R' in sum(original_rack, []) + sum(original_table, []):
            return '0_R'
        else:
            return '0_K'
    return tile


def insert_joker_correctly(set_group):
    if '0_K' in set_group:
        joker = '0_K'
    elif '0_R' in set_group:
        joker = '0_R'
    else:
        return set_group

    set_group.remove(joker)
    tiles = [(int(t.split('_')[0]), t) for t in set_group]
    tiles.sort()

    if tiles:
        for i in range(len(tiles) - 1):
            if tiles[i + 1][0] - tiles[i][0] > 1:
                tiles.insert(i + 1, (tiles[i][0] + 1, joker))
                break
        else:
            if tiles[0][0] > 1:
                tiles.insert(0, (1, joker))
            else:
                tiles.append((tiles[-1][0] + 1, joker))
    else:
        tiles.append((1, joker))

    return [t[1] for t in tiles]


def sort_new_table(table, new_table):
    table_groups = {tuple(group): i for i, group in enumerate(table)}
    new_table_sorted = []
    new_groups = []

    for group in new_table:
        if tuple(group) in table_groups:
            new_table_sorted.append((table_groups[tuple(group)], group))
        else:
            new_groups.append(group)

    new_table_sorted.sort(key=lambda x: x[0])
    new_table_sorted = [group for _, group in new_table_sorted]

    new_table_sorted.extend(new_groups)

    return new_table_sorted


def find_altered_groups(old_list, new_list):
    old_groups = {tuple(sorted(group)): i for i, group in enumerate(old_list)}
    new_groups = {tuple(sorted(group)): i for i, group in enumerate(new_list)}

    altered_groups = set()
    for group in new_groups:
        if group not in old_groups:
            altered_groups.add(new_groups[group])

    return altered_groups


def create_rummikub_grid(
    groups,
    altered_groups,
    highlighted_tiles,
    highlight_color,
    window_size,
    min_lines=4,
    is_rack=False
):
    padding = 25
    tile_size = (50, 80)
    max_tiles_per_line = (window_size[0] - 2 * padding) // tile_size[0]
    min_height_px = min_lines * tile_size[1]

    grid_image = np.zeros(
        (min_height_px, window_size[0] - 2 * padding, 3), dtype=np.uint8)
    x_offset, y_offset = 0, 0
    add_space = min_lines > 2
    tile_highlights = Counter(highlighted_tiles)

    highlight_boxes = []
    for group_index, group in enumerate(groups):
        if x_offset + len(group) > max_tiles_per_line:
            x_offset = 0
            y_offset += 1

        for tile in group:
            tile_image = cv2.imread(f'{tiles_path}/{tile}.png')
            if tile_image is not None:
                tile_image = cv2.resize(tile_image, tile_size)
                start_y = y_offset * tile_size[1]
                end_y = start_y + tile_size[1]
                start_x = x_offset * tile_size[0]
                end_x = start_x + tile_size[0]

                if end_y > grid_image.shape[0]:
                    extra_rows = np.zeros(
                        (tile_size[1], grid_image.shape[1], 3), dtype=np.uint8)
                    grid_image = np.vstack((grid_image, extra_rows))

                grid_image[start_y:end_y, start_x:end_x] = tile_image

                if is_rack:
                    if tile_highlights[tile] > 0:
                        highlight_boxes.append(
                            (start_x, start_y, end_x, end_y, highlight_color))
                        tile_highlights[tile] -= 1
                    else:
                        cv2.rectangle(
                            grid_image,
                            (start_x, start_y),
                            (end_x, end_y),
                            (255, 255, 255),
                            thickness=2
                        )
                else:
                    if group_index in altered_groups and tile_highlights[tile] > 0:
                        highlight_boxes.append(
                            (start_x, start_y, end_x, end_y, highlight_color))
                        tile_highlights[tile] -= 1
                    else:
                        cv2.rectangle(
                            grid_image,
                            (start_x, start_y),
                            (end_x, end_y),
                            (255, 255, 255),
                            thickness=2
                        )

            x_offset += 1

        if group_index < len(groups) - 1:
            if add_space:
                x_offset += 1
            if x_offset >= max_tiles_per_line:
                x_offset = 0
                y_offset += 1

    for box in highlight_boxes:
        cv2.rectangle(grid_image, (box[0], box[1]),
                      (box[2], box[3]), box[4], thickness=2)

    if grid_image.shape[0] > window_size[1] - 2 * padding or grid_image.shape[1] > window_size[0] - 2 * padding:
        grid_image = cv2.resize(
            grid_image,
            (window_size[0] - 2 * padding, window_size[1] - 2 * padding),
            interpolation=cv2.INTER_AREA
        )

    if grid_image.shape[0] < min_height_px:
        final_image = np.zeros(
            (min_height_px, window_size[0] - 2 * padding, 3), dtype=np.uint8)
        final_image[:grid_image.shape[0], :grid_image.shape[1]] = grid_image
    else:
        final_image = grid_image

    image_with_padding = cv2.copyMakeBorder(
        final_image,
        10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=[10, 10, 10]
    )
    final_image = cv2.copyMakeBorder(
        image_with_padding,
        15, 15, 15, 15,
        cv2.BORDER_CONSTANT,
        value=[40, 40, 40]
    )
    return final_image


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


async def process_game():
    sg = SetGenerator()
    tile_map, r_tile_map = create_number_maps(sg)

    table = get_tiles(table_image)
    print(f"Table: {table}")

    rack = get_tiles(rack_image)
    print(f"Rack: {rack}")

    table = [[replace_jokers(tile) for tile in set] for set in table]
    rack = [[replace_jokers(tile) for tile in set] for set in rack]

    numeric_table_example = [tile_map[tile]
                             for tile in sum(table, []) if tile in tile_map]
    numeric_rack_example = [tile_map[tile]
                            for tile in sum(rack, []) if tile in tile_map]

    solver = RummikubSolver(
        tiles=sg.tiles,
        sets=sg.sets,
        table=numeric_table_example,
        rack=numeric_rack_example
    )

    highlighted_tiles, new_table, remaining_rack, no_moves = get_solution(
        solver, r_tile_map, sum(rack, []))

    if not no_moves:
        highlighted_tiles = [revert_joker(t, table, rack)
                             for t in highlighted_tiles]
        new_table = [[revert_joker(t, table, rack)
                      for t in s] for s in new_table]
        new_table = [insert_joker_correctly(
            group) for group in new_table]
        new_table = sort_new_table(table, new_table)
        table = [[revert_joker(t, table, rack) for t in s] for s in table]
        rack = [[revert_joker(t, table, rack) for t in s] for s in rack]
        remaining_rack = [revert_joker(t, table, rack) for t in remaining_rack]

        print("Highlighted Tiles:", highlighted_tiles)
        print("New Table:", new_table)
        print("Remaining Rack:", remaining_rack)
        altered_groups = find_altered_groups(table, new_table)

        window_size = (940, 760)
        grid_image_top = create_rummikub_grid(
            table,
            altered_groups,
            highlighted_tiles,
            (245, 240, 0),
            window_size
        )
        grid_image_middle = create_rummikub_grid(
            new_table,
            altered_groups,
            highlighted_tiles,
            (0, 255, 0),
            window_size
        )
        grid_image_bottom = create_rummikub_grid(
            rack,
            altered_groups,
            highlighted_tiles,
            (245, 115, 255),
            window_size,
            min_lines=2,
            is_rack=True
        )
        grid_image_remaining = create_rummikub_grid(
            [[item] for item in remaining_rack],
            {},
            [],
            (255, 255, 255),
            window_size,
            min_lines=2
        )

        total_height = grid_image_top.shape[0] + grid_image_middle.shape[0] + \
            grid_image_bottom.shape[0] + grid_image_remaining.shape[0]
        combined_image = np.zeros(
            (total_height, window_size[0], 3), dtype=np.uint8)
        combined_image[:grid_image_top.shape[0], :] = grid_image_top
        current_y = grid_image_top.shape[0]
        combined_image[current_y:current_y +
                       grid_image_middle.shape[0], :] = grid_image_middle
        current_y += grid_image_middle.shape[0]
        combined_image[current_y:current_y +
                       grid_image_bottom.shape[0], :] = grid_image_bottom
        current_y += grid_image_bottom.shape[0]
        combined_image[current_y:, :] = grid_image_remaining

        IMAGE_PATH = os.path.join('game', 'rummikub.png')
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

        cv2.imwrite(IMAGE_PATH, combined_image)


async def handler(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'call_function':
                image_data = data['image'].split(",")[1]
                image_bytes = base64.b64decode(image_data)

                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.save(game_image)
                    crop_image(game_image, "table.png", 5, 28, 13, 12)
                    crop_image(game_image, "rack.png", 76.4, 1, 22.6, 22.9)
                print("Canvas image processed and saved.")

                await process_game()

                with Image.open("game/rummikub.png") as game:
                    with io.BytesIO() as output:
                        game.save(output, format='PNG')
                        binary_img = output.getvalue()
                        encoded_img = base64.b64encode(
                            binary_img).decode('utf-8')
                        await websocket.send(encoded_img)
                        print("Image 'rummikub.png' sent to client.")

    except Exception as e:
        print(f"WebSocket error: {e}")


async def main():
    start_server = websockets.serve(handler, "localhost", 5678, max_size=2**23)
    print("WebSocket server is running at ws://localhost:5678/")
    await start_server
    await asyncio.Future()

asyncio.run(main())

# tampermonkey

# // ==UserScript==
# // @name         Image Updater from WebSocket
# // @namespace    http://tampermonkey.net/
# // @version      0.1
# // @description  Update images from local WebSocket server
# // @author       You
# // @match        https://rummikub-apps.com/*
# // @grant        none
# // ==/UserScript==

# (function() {
#     'use strict';

#     document.getElementsByClassName('right-banner')[0].style.visibility = "hidden";
#     var ws = new WebSocket('ws://localhost:5678/');
#     ws.onopen = function() {
#         console.log("Conexão WebSocket estabelecida.");
#         var button = document.createElement("button");
#         button.innerHTML = ">";
#         button.style.position = "absolute";
#         button.style.top = "502px";
#         button.style.right = "543px";
#         button.style.zIndex = '1000';
#         button.style.fontWeight = "900";
#         button.style.fontFamily = "math";
#         button.style.backgroundColor = "transparent";
#         button.style.color = "white";
#         button.style.border = "1px solid wheat";
#         button.onclick = function() {
#             button.innerHTML = "✓";
#             setTimeout(function() {
#                 button.innerHTML = ">";
#             }, 2000);
#             requestAnimationFrame(() => {
#                 var canvas = document.getElementById('unity-canvas');
#                 if (canvas) {
#                     var image = canvas.toDataURL('image/png');
#                     console.log("Sending canvas image to server.");
#                     ws.send(JSON.stringify({type: 'call_function', image: image}));
#                 } else {
#                     console.log("Canvas element not found.");
#                 }
#             });
#         };
#         document.body.appendChild(button);
#     };

#     ws.onerror = function(error) {
#         console.log("WebSocket error:", error);
#     };

#     ws.onmessage = function(event) {
#         console.log("Image received from server");
#         var existingImg = document.getElementById('rummikub-img');
#         if (existingImg) {
#             document.body.removeChild(existingImg);
#         }

#         var img = new Image();
#         img.src = 'data:image/png;base64,' + event.data;
#         img.id = 'rummikub-img';
#         img.style.position = 'absolute';
#         img.style.top = '75px';
#         img.style.right = '40px';
#         img.style.zIndex = '1000';
#         img.style.width = '500px';
#         img.style.height = '680px';
#         document.body.appendChild(img);
#         var existingBtn = document.getElementById('hide-btn');
#         if (existingBtn) {
#             document.body.removeChild(existingBtn);
#         }

#         var button = document.createElement("button");
#         button.innerHTML = "^";
#         button.id = 'hide-btn';
#         button.style.position = 'absolute';
#         button.style.right = '544px';
#         button.style.bottom = '140px';
#         button.style.zIndex = '1001';
#         button.style.background = 'transparent';
#         button.style.border = '1px solid wheat';
#         button.style.color = 'white';
#         button.style.fontFamily = 'Math';
#         button.style.fontWeight = '900';
#         button.onclick = function() {
#             if (img.style.visibility === 'hidden') {
#                 img.style.visibility = 'visible';
#                 button.innerHTML = "^";
#             } else {
#                 img.style.visibility = 'hidden';
#                 button.innerHTML = "-";
#             }
#         };
#         img.parentNode.insertBefore(button, img.nextSibling);
#     };
# })();
