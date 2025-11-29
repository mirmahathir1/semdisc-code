from tdigest import TDigest

class SimpleTDigest:
    def __init__(self, data, compression=1):
        """Initialize the T-Digest with data and compression parameter."""
        self.tdigest = TDigest(delta=1/compression)
        self.tdigest.batch_update(data)
    
    def probability(self, x):
        """Return the approximate CDF value at point x."""
        return self.tdigest.cdf(x)
    
    # Additional useful methods from the tdigest library you might want to expose:
    def quantile(self, q):
        """Return the approximate quantile (inverse CDF)."""
        return self.tdigest.quantile(q)
    
    def add(self, x):
        """Add a single data point to the digest."""
        self.tdigest.update(x)
    
    def merge(self, other):
        """Merge another SimpleTDigest into this one."""
        self.tdigest += other.tdigest

if __name__ == "__main__":
    # Test with various input types
    samples_list = [1.2, 3.4, 2.1, 4.8, 3.3, 5.0, 2.9]
    
    print("Testing with list input:")
    cdf_list = SimpleTDigest(samples_list)
    print(f"P(X ≤ 5.1) = {cdf_list.probability(5.1)}")

