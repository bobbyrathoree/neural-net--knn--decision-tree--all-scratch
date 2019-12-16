from collections import defaultdict


def parse_image_data(file_path: str):
    data_vector = list()
    idx = 0
    images_counter = 0
    all_image_ids = defaultdict(tuple)

    file = open(file_path, "r")

    for image in file.read().split("\n"):
        if image == "":
            break

        each_line_split = image.split()
        all_image_ids[str(idx)] = (each_line_split[0], int(each_line_split[1]))
        data_vector.append([int(pixel_value) for pixel_value in each_line_split[2:]])

        idx += 1
        images_counter += 1

    file.close()

    return data_vector, all_image_ids, images_counter
