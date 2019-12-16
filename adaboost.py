from __future__ import division

from collections import defaultdict
import random
from copy import deepcopy
from math import log


class DecisionTreeStump(object):
    def __init__(self, first_pixel_index, second_pixel_index, alpha_value):
        self.first_pixel_index = first_pixel_index
        self.second_pixel_index = second_pixel_index
        self.alpha_value = alpha_value

    def get_properties(self):
        return self.first_pixel_index, self.second_pixel_index, self.alpha_value


class AdaBoost:
    def __init__(
        self,
        decision_stumps,
        images_data_vector: list = None,
        all_images_ids: defaultdict = None,
        images_counter: int = 0,
    ):
        self.decision_tree_stumps = decision_stumps
        self.all_images_ids = all_images_ids
        self.images_data_vector = images_data_vector
        self.total_images = images_counter
        self.combinations = set([])
        self.optimization_vector_size = [i for i in range(self.total_images)]
        self.confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        self.stumps_for_0_degrees, self.maximum_details_for_0_degrees, self.weights_for_0_degrees = (
            list(),
            (-1, 0, 0, 0),
            {
                str(index): 1 / self.total_images
                for index in self.optimization_vector_size
            },
        )
        self.stumps_for_90_degrees, self.maximum_details_for_90_degrees, self.weights_for_90_degrees = (
            list(),
            (-1, 0, 0, 0),
            deepcopy(self.weights_for_0_degrees),
        )
        self.stumps_for_180_degrees, self.maximum_details_for_180_degrees, self.weights_for_180_degrees = (
            list(),
            (-1, 0, 0, 0),
            deepcopy(self.weights_for_0_degrees),
        )
        self.stumps_for_270_degrees, self.maximum_details_for_270_degrees, self.weights_for_270_degrees = (
            list(),
            (-1, 0, 0, 0),
            deepcopy(self.weights_for_0_degrees),
        )

    def update_maximum_details(
        self,
        current_score,
        current_indexes,
        current_first_pixel,
        current_second_pixel,
        class_type,
    ):
        if class_type == 0:
            if current_score > self.maximum_details_for_0_degrees[0]:
                self.maximum_details_for_0_degrees = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )
        elif class_type == 90:
            if current_score > self.maximum_details_for_90_degrees[0]:
                self.maximum_details_for_90_degrees = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )
        elif class_type == 180:
            if current_score > self.maximum_details_for_180_degrees[0]:
                self.maximum_details_for_180_degrees = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )
        else:
            if current_score > self.maximum_details_for_270_degrees[0]:
                self.maximum_details_for_270_degrees = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )

    def get_maximum_details(self, first_position, second_position):
        classified_indexes_for_0_degrees, score_for_0_degrees = [], 0
        classified_indexes_for_90_degrees, score_for_90_degrees = [], 0
        classified_indexes_for_180_degrees, score_for_180_degrees = [], 0
        classified_indexes_for_270_degrees, score_for_270_degrees = [], 0

        for each_vector in self.optimization_vector_size:
            optimization_vector_as_string = str(each_vector)
            if (
                self.images_data_vector[each_vector][first_position]
                > self.images_data_vector[each_vector][second_position]
            ):
                if self.all_images_ids[optimization_vector_as_string][1] == 0:
                    score_for_0_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_0_degrees.append(
                        optimization_vector_as_string
                    )
                elif self.all_images_ids[optimization_vector_as_string][1] == 90:
                    score_for_90_degrees += self.weights_for_90_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_90_degrees.append(
                        optimization_vector_as_string
                    )
                elif self.all_images_ids[optimization_vector_as_string][1] == 180:
                    score_for_180_degrees += self.weights_for_180_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_180_degrees.append(
                        optimization_vector_as_string
                    )
                else:
                    score_for_270_degrees += self.weights_for_270_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_270_degrees.append(
                        optimization_vector_as_string
                    )
            else:
                if self.all_images_ids[str(each_vector)][1] == 0:
                    score_for_90_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_180_degrees += self.weights_for_180_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_270_degrees += self.weights_for_270_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_90_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_180_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_270_degrees.append(
                        optimization_vector_as_string
                    )
                elif self.all_images_ids[optimization_vector_as_string][1] == 90:
                    score_for_0_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_180_degrees += self.weights_for_180_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_270_degrees += self.weights_for_270_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_0_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_180_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_270_degrees.append(
                        optimization_vector_as_string
                    )
                elif self.all_images_ids[str(each_vector)][1] == 180:
                    score_for_0_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_90_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_270_degrees += self.weights_for_270_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_0_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_90_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_270_degrees.append(
                        optimization_vector_as_string
                    )
                else:
                    score_for_0_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_90_degrees += self.weights_for_0_degrees[
                        optimization_vector_as_string
                    ]
                    score_for_180_degrees += self.weights_for_180_degrees[
                        optimization_vector_as_string
                    ]
                    classified_indexes_for_0_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_90_degrees.append(
                        optimization_vector_as_string
                    )
                    classified_indexes_for_180_degrees.append(
                        optimization_vector_as_string
                    )

        self.update_maximum_details(
            score_for_0_degrees,
            classified_indexes_for_0_degrees,
            first_position,
            second_position,
            0,
        )
        self.update_maximum_details(
            score_for_90_degrees,
            classified_indexes_for_90_degrees,
            first_position,
            second_position,
            90,
        )
        self.update_maximum_details(
            score_for_180_degrees,
            classified_indexes_for_180_degrees,
            first_position,
            second_position,
            180,
        )
        self.update_maximum_details(
            score_for_270_degrees,
            classified_indexes_for_270_degrees,
            first_position,
            second_position,
            270,
        )

    def set_initial_maximum_details(self):
        self.maximum_details_for_0_degrees = (-1, 0, 0, 0)
        self.maximum_details_for_90_degrees = (-1, 0, 0, 0)
        self.maximum_details_for_180_degrees = (-1, 0, 0, 0)
        self.maximum_details_for_270_degrees = (-1, 0, 0, 0)

    def append_best_stump(
        self,
        best_indexes_for_0_degrees,
        best_indexes_for_90_degrees,
        best_indexes_for_180_degrees,
        best_indexes_for_270_degrees,
        alpha_value_for_0_degrees,
        alpha_value_for_90_degrees,
        alpha_value_for_180_degrees,
        alpha_value_for_270_degrees,
    ):
        self.stumps_for_0_degrees.append(
            self.get_stump(best_indexes_for_0_degrees, alpha_value_for_0_degrees)
        )
        self.stumps_for_90_degrees.append(
            self.get_stump(best_indexes_for_90_degrees, alpha_value_for_90_degrees)
        )
        self.stumps_for_180_degrees.append(
            self.get_stump(best_indexes_for_180_degrees, alpha_value_for_180_degrees)
        )
        self.stumps_for_270_degrees.append(
            self.get_stump(best_indexes_for_270_degrees, alpha_value_for_270_degrees)
        )

    def normalize_weights(self):
        total_0 = sum(self.weights_for_0_degrees.values())
        total_90 = sum(self.weights_for_90_degrees.values())
        total_180 = sum(self.weights_for_180_degrees.values())
        total_270 = sum(self.weights_for_270_degrees.values())

        for each_vector in self.optimization_vector_size:
            self.weights_for_0_degrees[str(each_vector)] /= total_0
            self.weights_for_90_degrees[str(each_vector)] /= total_90
            self.weights_for_180_degrees[str(each_vector)] /= total_180
            self.weights_for_270_degrees[str(each_vector)] /= total_270

    def update_weights(
        self,
        maximum_details_for_0_degrees,
        maximum_details_for_90_degrees,
        maximum_details_for_180_degrees,
        maximum_details_for_270_degrees,
        beta_value_for_0_degrees,
        beta_value_for_90_degrees,
        beta_value_for_180_degrees,
        beta_value_for_270_degrees,
    ):
        maximum_details_for_0_degrees = set(maximum_details_for_0_degrees)
        maximum_details_for_90_degrees = set(maximum_details_for_90_degrees)
        maximum_details_for_180_degrees = set(maximum_details_for_180_degrees)
        maximum_details_for_270_degrees = set(maximum_details_for_270_degrees)

        for each_vector in self.optimization_vector_size:
            optimization_vector_as_string = str(each_vector)

            current_weight_for_0_degrees = self.weights_for_0_degrees[
                optimization_vector_as_string
            ]
            self.weights_for_0_degrees[optimization_vector_as_string] = (
                current_weight_for_0_degrees * beta_value_for_0_degrees
                if optimization_vector_as_string in maximum_details_for_0_degrees
                else current_weight_for_0_degrees
            )

            current_weight_for_90_degrees = self.weights_for_90_degrees[
                optimization_vector_as_string
            ]
            self.weights_for_90_degrees[optimization_vector_as_string] = (
                current_weight_for_90_degrees * beta_value_for_90_degrees
                if optimization_vector_as_string in maximum_details_for_90_degrees
                else current_weight_for_90_degrees
            )

            current_weight_for_180_degrees = self.weights_for_180_degrees[
                optimization_vector_as_string
            ]
            self.weights_for_180_degrees[optimization_vector_as_string] = (
                current_weight_for_180_degrees * beta_value_for_180_degrees
                if optimization_vector_as_string in maximum_details_for_180_degrees
                else current_weight_for_180_degrees
            )

            current_weight_for_270_degrees = self.weights_for_270_degrees[
                optimization_vector_as_string
            ]
            self.weights_for_270_degrees[optimization_vector_as_string] = (
                current_weight_for_270_degrees * beta_value_for_270_degrees
                if optimization_vector_as_string in maximum_details_for_270_degrees
                else current_weight_for_270_degrees
            )

        self.normalize_weights()

    def infer_orientation(self, testing_image):
        orientation = 0
        vote = self.get_positive_votes(testing_image, self.stumps_for_0_degrees)

        next_vote = self.get_positive_votes(testing_image, self.stumps_for_90_degrees)
        if next_vote > vote:
            orientation = 90
            vote = next_vote

        next_vote = self.get_positive_votes(testing_image, self.stumps_for_180_degrees)
        if next_vote > vote:
            orientation = 180
            vote = next_vote

        next_vote = self.get_positive_votes(testing_image, self.stumps_for_270_degrees)
        if next_vote > vote:
            orientation = 270
            vote = next_vote

        return orientation

    def set_random_combinations(self):
        for i in range(1000):
            first_idx, second_idx = random.randint(0, 191), random.randint(0, 191)
            while (
                first_idx == second_idx or (first_idx, second_idx) in self.combinations
            ):
                first_idx, second_idx = random.randint(0, 191), random.randint(0, 191)
            self.combinations.add((first_idx, second_idx))

    def test(self, test_file_path):
        file = open(test_file_path, "r")
        for image in file.read().split("\n"):
            if image == "":
                break
            each_line_split = image.split()
            testing_image = [int(pixel) for pixel in each_line_split[2:]]
            inferred_orientation = self.infer_orientation(testing_image)

            self.confusion_matrix[int(int(each_line_split[1]) / 90)][
                int(inferred_orientation / 90)
            ] += 1

            print(
                "Found orientation for: ",
                each_line_split[0],
                ": ",
                str(inferred_orientation),
                "Original orientation (given in Train): ",
                each_line_split[1],
            )
        file.close()
        print(
            "\nAccuracy Percentage: {0}%".format(
                round(
                    sum([self.confusion_matrix[i][i] for i in range(4)])
                    * 100.0
                    / sum(
                        [
                            self.confusion_matrix[i][j]
                            for i in range(4)
                            for j in range(4)
                        ]
                    ),
                    2,
                )
            )
        )

    def train(self):
        self.set_random_combinations()

        for i in range(self.decision_tree_stumps):
            print(i, "stump")
            self.set_initial_maximum_details()

            for indexes in self.combinations:
                self.get_maximum_details(*indexes)

            beta_value_for_0_degrees, beta_value_for_90_degrees, beta_value_for_180_degrees, beta_value_for_270_degrees = map(
                self.get_beta,
                [
                    self.maximum_details_for_0_degrees[0],
                    self.maximum_details_for_90_degrees[0],
                    self.maximum_details_for_180_degrees[0],
                    self.maximum_details_for_270_degrees[0],
                ],
            )
            alpha_value_for_0_degrees, alpha_value_for_90_degrees, alpha_value_for_180_degrees, alpha_value_for_270_degrees = map(
                self.get_alpha,
                [
                    beta_value_for_0_degrees,
                    beta_value_for_90_degrees,
                    beta_value_for_180_degrees,
                    beta_value_for_270_degrees,
                ],
            )
            self.append_best_stump(
                self.maximum_details_for_0_degrees[2:4],
                self.maximum_details_for_90_degrees[2:4],
                self.maximum_details_for_180_degrees[2:4],
                self.maximum_details_for_270_degrees[2:4],
                alpha_value_for_0_degrees,
                alpha_value_for_90_degrees,
                alpha_value_for_180_degrees,
                alpha_value_for_270_degrees,
            )

            self.update_weights(
                self.maximum_details_for_0_degrees[1],
                self.maximum_details_for_90_degrees[1],
                self.maximum_details_for_180_degrees[1],
                self.maximum_details_for_270_degrees[1],
                beta_value_for_0_degrees,
                beta_value_for_90_degrees,
                beta_value_for_180_degrees,
                beta_value_for_270_degrees,
            )

    @staticmethod
    def get_beta(maximum_detail_score):
        return (1 - maximum_detail_score) / maximum_detail_score

    @staticmethod
    def get_alpha(beta_value):
        return log(1 / beta_value)

    @staticmethod
    def get_positive_votes(test_image, decision_tree_stumps):
        p_votes = 0
        for one_stump in decision_tree_stumps:
            first_pixel_index, second_pixel_index, alpha_value = (
                one_stump.get_properties()
            )
            if test_image[first_pixel_index] > test_image[second_pixel_index]:
                p_votes += alpha_value
        return p_votes

    @staticmethod
    def get_stump(indexes, weight):
        return DecisionTreeStump(indexes[0], indexes[1], weight)
