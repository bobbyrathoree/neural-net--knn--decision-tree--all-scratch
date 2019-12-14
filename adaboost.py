from collections import defaultdict
from copy import deepcopy
import random
from math import log


class DecisionTreeStump(object):
    def __init__(self, first_pixel_index, second_pixel_index, alpha_value):
        self.first_pixel_index = first_pixel_index
        self.second_pixel_index = second_pixel_index
        self.alpha_value = alpha_value

    def get_properties(self):
        return self.first_pixel_index, self.second_pixel_index, self.alpha_value


class AdaBoost(object):
    def __init__(
        self,
        images_data_vector: list,
        all_images_ids: defaultdict,
        images_counter: int,
        decision_stumps: int = 30,
    ):
        self.decision_tree_stumps = decision_stumps
        self.images_data_vector = images_data_vector
        self.all_images_ids = all_images_ids
        self.total_images = images_counter
        self.combinations = set(tuple())
        self.optimization_vector_size = [i for i in range(self.total_images)]

        self.stumps_for_0, self.maximum_details_for_0, self.weights_for_0 = (
            list(),
            (-1, 0, 0, 0),
            {
                str(index): 1 / self.total_images
                for index in self.optimization_vector_size
            },
        )
        self.stumps_for_90, self.maximum_details_for_90, self.weights_for_90 = (
            list(),
            (-1, 0, 0, 0),
            deepcopy(self.weights_for_0),
        )
        self.stumps_for_180, self.maximum_details_for_180, self.weights_for_180 = (
            list(),
            (-1, 0, 0, 0),
            deepcopy(self.weights_for_0),
        )
        self.stumps_for_270, self.maximum_details_for_270, self.weights_for_270 = (
            list(),
            (-1, 0, 0, 0),
            deepcopy(self.weights_for_0),
        )

    def train(self):
        self.set_random_combinations()

        for i in range(self.decision_tree_stumps):
            for one_combination_of_index in self.combinations:
                self.get_maximum_details(*one_combination_of_index)

            beta_value_for_0, beta_value_for_90, beta_value_for_180, beta_value_for_270 = map(
                self.get_beta,
                [
                    self.maximum_details_for_0[0],
                    self.maximum_details_for_90[0],
                    self.maximum_details_for_180[0],
                    self.maximum_details_for_270[0],
                ],
            )

            alpha0, alpha90, alpha180, alpha270 = map(
                self.get_alpha,
                [
                    beta_value_for_0,
                    beta_value_for_90,
                    beta_value_for_180,
                    beta_value_for_270,
                ],
            )

            self.append_best_stump(
                self.maximum_details_for_0[2:4],
                self.maximum_details_for_90[2:4],
                self.maximum_details_for_180[2:4],
                self.maximum_details_for_270[2:4],
                alpha0,
                alpha90,
                alpha180,
                alpha270,
            )

            self.update_weights(
                self.maximum_details_for_0[1],
                self.maximum_details_for_90[1],
                self.maximum_details_for_180[1],
                self.maximum_details_for_270[1],
                beta_value_for_0,
                beta_value_for_90,
                beta_value_for_180,
                beta_value_for_270,
            )

            self.normalize_weights()

    def set_random_combinations(self):
        for i in range(2500):
            first_idx, second_idx = random.randint(0, 191), random.randint(0, 191)
            while (
                first_idx == second_idx or (first_idx, second_idx) in self.combinations
            ):
                first_idx, second_idx = random.randint(0, 191), random.randint(0, 191)
            self.combinations.add((first_idx, second_idx))

    def get_maximum_details(self, first_position, second_position):
        # Store the indexes of the images which were correctly classified
        classified_indexes_for_0, score_for_0 = [], 0
        classified_indexes_for_90, score_for_90 = [], 0
        classified_indexes_for_180, score_for_180 = [], 0
        classified_indexes_for_270, score_for_270 = [], 0

        for each_vector in self.optimization_vector_size:
            optimization_vector_as_string = str(each_vector)
            if (
                self.images_data_vector[each_vector][first_position]
                > self.images_data_vector[each_vector][second_position]
            ):
                if self.all_images_ids[optimization_vector_as_string][1] == 0:
                    score_for_0 += self.weights_for_0[optimization_vector_as_string]
                    classified_indexes_for_0.append(optimization_vector_as_string)
                elif self.all_images_ids[optimization_vector_as_string][1] == 90:
                    score_for_90 += self.weights_for_90[optimization_vector_as_string]
                    classified_indexes_for_90.append(optimization_vector_as_string)
                elif self.all_images_ids[optimization_vector_as_string][1] == 180:
                    score_for_180 += self.weights_for_180[optimization_vector_as_string]
                    classified_indexes_for_180.append(optimization_vector_as_string)
                else:
                    score_for_270 += self.weights_for_270[optimization_vector_as_string]
                    classified_indexes_for_270.append(optimization_vector_as_string)
            else:
                if self.all_images_ids[str(each_vector)][1] == 0:
                    score_for_90 += self.weights_for_0[optimization_vector_as_string]
                    score_for_180 += self.weights_for_180[optimization_vector_as_string]
                    score_for_270 += self.weights_for_270[optimization_vector_as_string]
                    classified_indexes_for_90.append(optimization_vector_as_string)
                    classified_indexes_for_180.append(optimization_vector_as_string)
                    classified_indexes_for_270.append(optimization_vector_as_string)
                elif self.all_images_ids[optimization_vector_as_string][1] == 90:
                    score_for_0 += self.weights_for_0[optimization_vector_as_string]
                    score_for_180 += self.weights_for_180[optimization_vector_as_string]
                    score_for_270 += self.weights_for_270[optimization_vector_as_string]
                    classified_indexes_for_0.append(optimization_vector_as_string)
                    classified_indexes_for_180.append(optimization_vector_as_string)
                    classified_indexes_for_270.append(optimization_vector_as_string)
                elif self.all_images_ids[str(each_vector)][1] == 180:
                    score_for_0 += self.weights_for_0[optimization_vector_as_string]
                    score_for_90 += self.weights_for_0[optimization_vector_as_string]
                    score_for_270 += self.weights_for_270[optimization_vector_as_string]
                    classified_indexes_for_0.append(optimization_vector_as_string)
                    classified_indexes_for_90.append(optimization_vector_as_string)
                    classified_indexes_for_270.append(optimization_vector_as_string)
                else:
                    score_for_0 += self.weights_for_0[optimization_vector_as_string]
                    score_for_90 += self.weights_for_0[optimization_vector_as_string]
                    score_for_180 += self.weights_for_180[optimization_vector_as_string]
                    classified_indexes_for_0.append(optimization_vector_as_string)
                    classified_indexes_for_90.append(optimization_vector_as_string)
                    classified_indexes_for_180.append(optimization_vector_as_string)

        self.update_maximum_details(
            score_for_0, classified_indexes_for_0, first_position, second_position, 0
        )
        self.update_maximum_details(
            score_for_90, classified_indexes_for_90, first_position, second_position, 90
        )
        self.update_maximum_details(
            score_for_180,
            classified_indexes_for_180,
            first_position,
            second_position,
            180,
        )
        self.update_maximum_details(
            score_for_270,
            classified_indexes_for_270,
            first_position,
            second_position,
            270,
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
            if current_score > self.maximum_details_for_0[0]:
                self.maximum_details_for_0 = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )
        elif class_type == 90:
            if current_score > self.maximum_details_for_90[0]:
                self.maximum_details_for_90 = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )
        elif class_type == 180:
            if current_score > self.maximum_details_for_180[0]:
                self.maximum_details_for_180 = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )
        else:
            if current_score > self.maximum_details_for_270[0]:
                self.maximum_details_for_270 = (
                    current_score,
                    current_indexes,
                    current_first_pixel,
                    current_second_pixel,
                )

    def append_best_stump(
        self,
        best_indexes_for_0,
        best_indexes_for_90,
        best_indexes_for_180,
        best_indexes_for_270,
        alpha_value_for_0,
        alpha_value_for_90,
        alpha_value_for_180,
        alpha_value_for_270,
    ):
        self.stumps_for_0.append(self.get_stump(best_indexes_for_0, alpha_value_for_0))
        self.stumps_for_90.append(
            self.get_stump(best_indexes_for_90, alpha_value_for_90)
        )
        self.stumps_for_180.append(
            self.get_stump(best_indexes_for_180, alpha_value_for_180)
        )
        self.stumps_for_270.append(
            self.get_stump(best_indexes_for_270, alpha_value_for_270)
        )

    def update_weights(
        self,
        maximum_details_for_0,
        maximum_details_for_90,
        maximum_details_for_180,
        maximum_details_for_270,
        beta_value_for_0,
        beta_value_for_90,
        beta_value_for_180,
        beta_value_for_270,
    ):
        maximum_details_for_0 = set(maximum_details_for_0)
        maximum_details_for_90 = set(maximum_details_for_90)
        maximum_details_for_180 = set(maximum_details_for_180)
        maximum_details_for_270 = set(maximum_details_for_270)

        for each_vector in self.optimization_vector_size:
            optimization_vector_as_string = str(each_vector)

            current_weight_for_0 = self.weights_for_0[optimization_vector_as_string]
            self.weights_for_0[optimization_vector_as_string] = (
                current_weight_for_0 * beta_value_for_0
                if optimization_vector_as_string in maximum_details_for_0
                else current_weight_for_0
            )

            current_weight_for_90 = self.weights_for_90[optimization_vector_as_string]
            self.weights_for_90[optimization_vector_as_string] = (
                current_weight_for_90 * beta_value_for_90
                if optimization_vector_as_string in maximum_details_for_90
                else current_weight_for_90
            )

            current_weight_for_180 = self.weights_for_180[optimization_vector_as_string]
            self.weights_for_180[optimization_vector_as_string] = (
                current_weight_for_180 * beta_value_for_180
                if optimization_vector_as_string in maximum_details_for_180
                else current_weight_for_180
            )

            current_weight_for_270 = self.weights_for_270[optimization_vector_as_string]
            self.weights_for_270[optimization_vector_as_string] = (
                current_weight_for_270 * beta_value_for_270
                if optimization_vector_as_string in maximum_details_for_270
                else current_weight_for_270
            )

    def normalize_weights(self):
        total_0 = sum(self.weights_for_0.values())
        total_90 = sum(self.weights_for_90.values())
        total_180 = sum(self.weights_for_180.values())
        total_270 = sum(self.weights_for_270.values())

        for each_vector in self.optimization_vector_size:
            self.weights_for_0[str(each_vector)] /= total_0
            self.weights_for_90[str(each_vector)] /= total_90
            self.weights_for_180[str(each_vector)] /= total_180
            self.weights_for_270[str(each_vector)] /= total_270

    @staticmethod
    def get_beta(maximum_detail_score):
        return (1 - maximum_detail_score) / maximum_detail_score

    @staticmethod
    def get_alpha(beta_value):
        return log(1 / beta_value)

    @staticmethod
    def get_stump(indexes, weight):
        return DecisionTreeStump(indexes[0], indexes[1], weight)
