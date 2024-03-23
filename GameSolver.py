def can_add_tile_to_group(group, tile):
    tile_number, tile_color = tile.split('_')
    group_numbers = [t.split('_')[0] for t in group]
    group_colors = [t.split('_')[1] for t in group]

    if all(g == group_colors[0] for g in group_colors):
        if tile_color == group_colors[0]:
            if str(int(group_numbers[-1]) + 1) == tile_number or str(int(group_numbers[0]) - 1) == tile_number:
                return True
    else:
        if tile_number in group_numbers and tile_color not in group_colors:
            return True
    return False


def form_new_groups(rack):
    new_groups = []
    tiles_by_number = {}
    for tile in rack:
        number, color = tile.split('_')
        if number not in tiles_by_number:
            tiles_by_number[number] = []
        tiles_by_number[number].append(tile)
    for number, tiles in tiles_by_number.items():
        if len(tiles) >= 3:
            new_groups.append(tiles)
            for tile in tiles:
                rack.remove(tile)
    return new_groups


def form_new_sequences(rack):
    new_sequences = []
    tiles_by_color = {}
    for tile in rack:
        number, color = tile.split('_')
        if color not in tiles_by_color:
            tiles_by_color[color] = []
        tiles_by_color[color].append((int(number), tile))
    for color, numbered_tiles in tiles_by_color.items():
        numbered_tiles.sort()
        sequence = [numbered_tiles[0][1]]
        for i in range(1, len(numbered_tiles)):
            if numbered_tiles[i][0] == numbered_tiles[i-1][0] + 1:
                sequence.append(numbered_tiles[i][1])
            else:
                if len(sequence) >= 3:
                    new_sequences.append(sequence)
                    sequence = [numbered_tiles[i][1]]
                else:
                    sequence = [numbered_tiles[i][1]]
        if len(sequence) >= 3:
            new_sequences.append(sequence)
    for seq in new_sequences:
        for tile in seq:
            if tile in rack:
                rack.remove(tile)
    return new_sequences


def find_valid_play(table, rack):
    initial_rack_size = len(rack)
    for tile in list(rack):
        for group in table:
            if can_add_tile_to_group(group, tile):
                group.append(tile)
                rack.remove(tile)
                break

    if len(rack) < initial_rack_size:
        if not rack:
            return table
        else:
            new_groups = form_new_groups(rack)
            new_sequences = form_new_sequences(rack)
            if new_groups or new_sequences:
                table.extend(new_groups + new_sequences)

    return table if not rack else [["buy new tile"]]


table_example = [["1_B", "2_B", "3_B"], ["6_R", "6_K", "6_Y"]]
rack_example = ["4_B", "7_R", "2_R", "3_R", "4_R"]
new_table = find_valid_play(table_example, rack_example)
print(new_table)
