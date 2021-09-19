import numpy as np


class Kmeans:
    # Initialize with number of clusters and an integer n for random initilialization
    def __init__(self, n_clusters, n):
        self.m = {i: [] for i in range(n_clusters)}
        self.centroid = np.random.randint(low=0, high=n, size=(n_clusters, 2))
        self.n_clusters = n_clusters

    # Fit with data x
    def fit(self, x):
        n = len(x)
        first = True
        while True:
            for i in range(n):
                result = map(lambda y: np.linalg.norm(y - x[i]), self.centroid)
                dist = list(result)
                self.m[dist.index(min(dist))].append(x[i])

            if first:
                prev_centroid = self.centroid
                for i in range(len(self.centroid)):
                    self.centroid[i] = np.average(self.m[i], axis=0)
                first = False
                self.m = {i: [] for i in range(self.n_clusters)}
                continue
            abs_diff = np.absolute(prev_centroid - self.centroid)
            diff_avg = np.average(abs_diff, axis=0)

            if diff_avg[0] < 1e-7 and diff_avg[1] < 1e-7:
                break
            else:
                prev_centroid = self.centroid
                for i in range(len(self.centroid)):
                    self.centroid[i] = np.average(self.m[i], axis=0)
                self.m = {i: [] for i in range(self.n_clusters)}
                continue

    # Predict for sample x
    def predict(self, x):
        dist_index = []
        for i in range(len(x)):
            result = map(lambda y: np.linalg.norm(y - x[i]), self.centroid)
            dist = list(result)
            dist_index.append(dist.index(min(dist)))
        return dist_index

    # Fit and predict for data x
    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    # Transform data to distance with each cluser space
    def transform(self, x):
        dist = []
        for i in range(len(x)):
            result = map(lambda y: np.linalg.norm(y - x[i]), self.centroid)
            d = list(result)
            dist.append(d)
        return dist

    # Fit and transform for data x
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


if __name__ == '__main__':
    a = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    km = Kmeans(n_clusters=2, n=len(a))
    # km.fit(a)
    print(km.fit_transform(a))
    # print(km.fit_predict([[0, 2]]))
