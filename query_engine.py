import numpy as np


class SearchEngine:
    def __init__(self, index):
        # store index of searched images
        self.index = index

    def engine(self, queryImageFeatures):
        results = {}

        # looping over index file

        for (filename, features) in self.index.items():
            # chi-squared distance which is normally used in the
            # computer vision field to compare histograms
            distance = self.chi2_distance(features, queryImageFeatures)
            
			# the representing the similarity of the images in index
			
            results[filename] = distance

            # sorting the relevent images
			
        results = sorted([(features, filename) for (filename, features) in results.items()])
        return results
  
    def chi2_distance(hist1, hist2, eps=1e-10):
        distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(hist1, hist2)])

        return distance