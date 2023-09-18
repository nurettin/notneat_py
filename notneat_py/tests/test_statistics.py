import unittest
from typing import List, Callable, Any, Tuple, Optional

from notneat_py.statistics import group_by_standard_deviation, rank_by_standard_deviation


class TestRankingAndGrouping(unittest.TestCase):

    def test_rank_by_standard_deviation(self):
        objects = [1, 2, 3, 4, 5]
        distance_function = lambda x, y: abs(x - y)
        ranks = rank_by_standard_deviation(objects, distance_function)
        
        # Ensure the list is sorted by standard deviation in descending order
        self.assertTrue(all(ranks[i][1] >= ranks[i + 1][1] for i in range(len(ranks) - 1)))

    def test_group_by_standard_deviation(self):
        ranks = [(5, 10), (4, 8), (3, 5), (2, 4), (1, 1)]
        groups = group_by_standard_deviation(ranks)
        
        # Calculate the median gap
        gaps = [abs(ranks[i][1] - ranks[i - 1][1]) for i in range(1, len(ranks))]
        gaps.sort()
        median_index = len(gaps) // 2
        threshold = gaps[median_index] if len(gaps) % 2 == 1 else (gaps[median_index - 1] + gaps[median_index]) / 2
        
        # Test that objects are grouped correctly
        group_id = 0
        for i in range(1, len(ranks)):
            if abs(ranks[i][1] - ranks[i - 1][1]) > threshold:
                group_id += 1
            self.assertIn(ranks[i], groups[group_id]) # Assert that each rank is in the correct group

if __name__ == '__main__':
    unittest.main()
