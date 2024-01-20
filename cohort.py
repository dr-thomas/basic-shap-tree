import itertools
import math
import numpy as np

class Sample:
    def __init__(self) -> None:
        self.features: dict[int, float] = {} # no 'any' for typing, but categorical data need not be encoded
        self.label: int = -999

class Cohort:
    def __init__(self) -> None:
        self.samples: dict[int, Sample] = {}
        self.is_ftr_categorical: dict[int, bool] = {}
        self.feature_names: dict[int, str] = {}
        self.cat_ftr_groupings: dict[int, dict[int, list[int]]] = {}
        self.feature_threshs: dict[int, list[float]] = {}

    def calc_categorical_feature_groupings(self) -> None:
        for ftr_idx in self.is_ftr_categorical:
            if self.is_ftr_categorical[ftr_idx]:
                cats = []
                for sample_idx in self.samples:
                    cats.append(self.samples[sample_idx].features[ftr_idx])
                cats = list(set(cats))
                n_cats = len(cats)
                self.cat_ftr_groupings[ftr_idx] = {}
                grouping_idx = 0
                if n_cats%2 == 0:
                    for idraw in range(1, int(n_cats/2)):
                        for combo in itertools.combinations(cats, idraw):
                            self.cat_ftr_groupings[ftr_idx][grouping_idx] = list(combo)
                            grouping_idx += 1
                    n_combos = math.comb(n_cats, int(n_cats)/2)
                    for icombo, combo in enumerate(itertools.combinations(cats, int(n_cats)/2)):
                        if icombo < int(n_combos/2):
                            self.cat_ftr_groupings[ftr_idx][grouping_idx] = list(combo)
                            grouping_idx += 1
                else:
                    for idraw in range(1, int((n_cats+1)/2)):
                        for combo in itertools.combinations(cats, idraw):
                            self.cat_ftr_groupings[ftr_idx][grouping_idx] = list(combo)
                            grouping_idx += 1

    def calc_feature_thresholds(self, n_bins: int = -999) -> None:
        if n_bins < 0:
            n_bins = len(self.samples)
        for ftr_idx in self.is_ftr_categorical:
            if not self.is_ftr_categorical[ftr_idx]:
                ftr_vals = []
                for sample_idx in self.samples:
                    ftr_vals.append(self.samples[sample_idx].features[ftr_idx])
                self.feature_threshs[ftr_idx] = []
                for ibin in range(n_bins):
                    self.feature_threshs[ftr_idx].append(np.percentile(ftr_vals, 100*(ibin+1)/n_bins))