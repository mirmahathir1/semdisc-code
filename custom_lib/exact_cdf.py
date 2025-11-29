import numpy as np
import bisect

class ExactCDF:
    def __init__(self, data):
        """Initialize with sorted data and precompute weights."""
        self.sorted_data = np.sort(data)
        self.n = len(self.sorted_data)
        self.total_weight = self.n
    
    def probability(self, x):
        """Compute exact CDF using empirical distribution function."""
        return self._single_probability(x)
    
    def _single_probability(self, x):
        # Find the first index where data >= x
        idx = bisect.bisect_left(self.sorted_data, x)
        
        # The CDF at x is the proportion of values <= x
        return idx / self.n

if __name__ == "__main__":
    # Test with various input types
    samples_list = [1.2, 3.4, 2.1, 4.8, 3.3, 5.0, 2.9]
    
    print("Testing with list input:")
    exact_cdf = ExactCDF(samples_list)
    print(f"P(X ≤ 3.0) = {exact_cdf.probability(3.0)}")
    print(f"P(X ≤ 2.1) = {exact_cdf.probability(2.1)}")
    print(f"P(X ≤ 5.1) = {exact_cdf.probability(5.1)}")
    print(f"P(X ≤ 0.0) = {exact_cdf.probability(0.0)}")
    
    # Test with array input
    test_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print("\nTesting with array input:")
    print(f"CDF values: {exact_cdf.probability(test_points)}")