import cohort
import random
import math
import itertools
import matplotlib.pyplot as plt
import scipy.stats

class Node:
    def __init__(self) -> None:
        self.child_left: int = -999
        self.child_right: int = -999
        self.val: int = -999
        self.ftr_idx: int = -999
        self.thresh: float = -999.9
        self.is_categorical: bool = False
        self.left_cat_group_idxs: list[int] = []

    def predict(self, sample: cohort.Sample) -> int:
        if not self.ftr_idx in sample.features:
            return -999 # eject if missing feature
        if self.is_categorical:
            if sample.features[self.ftr_idx] in self.left_cat_group_idxs:
                return self.child_left
        else:
            if sample.features[self.ftr_idx] < self.thresh:
                return self.child_left
        return self.child_right

class TrainingNode(Node):
    def __init__(self) -> None:
        super().__init__()
        self.enropy: float = -999.9
        self.sample_idxs: list[int] = []
        self.depth: int = -999

class Tree(cohort.Cohort):
    def __init__(self, training_cohort: cohort.Cohort) -> None:
        self.samples = training_cohort.samples
        self.is_ftr_categorical = training_cohort.is_ftr_categorical
        self.feature_names = training_cohort.feature_names
        self.cat_ftr_groupings = training_cohort.cat_ftr_groupings
        self.feature_threshs = training_cohort.feature_threshs

        self.nodes: list[TrainingNode] = []
        self.use_ftr_idxs: list[int] = [idx for idx in self.is_ftr_categorical]
        self.nftrs_to_sample: int = int(math.sqrt(len(self.use_ftr_idxs)))
        self.max_depth: int = 100
        self.min_leaf_size: int = 1

    def split_node(self, node_idx: int) -> None:
        # TODO: need stopping criteria: max depth, min_leaf_size
        if self.nodes[node_idx].depth > self.max_depth:
            return
        ftr_idxs = random.sample(self.use_ftr_idxs, self.nftrs_to_sample)
        max_ftr_idx = -1
        max_group_idx = -1
        max_thresh = -999.9
        max_info_gain = -999.9
        for ftr_idx in ftr_idxs:
            if self.is_ftr_categorical[ftr_idx]:
                for group_idx in self.cat_ftr_groupings[ftr_idx]:
                    info_gain = self.calc_info_gain_cat(ftr_idx=ftr_idx, group_idx=group_idx, node_idx=node_idx)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        max_ftr_idx = ftr_idx
                        max_group_idx = group_idx
            else:
                for thresh in self.feature_threshs[ftr_idx]:
                    info_gain = self.calc_info_gain(ftr_idx=ftr_idx, thresh=thresh, node_idx=node_idx)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        max_ftr_idx = ftr_idx
                        max_thresh = thresh
        
        if max_ftr_idx < 0 or max_info_gain < 0:
            return
        
        self.nodes[node_idx].ftr_idx = max_ftr_idx
        
        if self.is_ftr_categorical[max_ftr_idx]:
            left_idxs = []
            right_idxs = []
            for sample_idx in self.nodes[node_idx].sample_idxs:
                if not max_ftr_idx in self.samples[sample_idx].features:
                    continue
                if self.samples[sample_idx].features[max_ftr_idx] in self.cat_ftr_groupings[max_ftr_idx][max_group_idx]:
                    left_idxs.append(sample_idx)
                else:
                    right_idxs.append(sample_idx)
            self.nodes[node_idx].is_categorical = True
            self.nodes[node_idx].left_cat_group_idxs = self.cat_ftr_groupings[max_ftr_idx][max_group_idx]
        else:
            left_idxs = []
            right_idxs = []
            for sample_idx in self.nodes[node_idx].sample_idxs:
                if not max_ftr_idx in self.samples[sample_idx].features:
                    continue
                if self.samples[sample_idx].features[max_ftr_idx] < max_thresh:
                    left_idxs.append(sample_idx)
                else:
                    right_idxs.append(sample_idx)
            self.nodes[node_idx].thresh = max_thresh
        
        child_left = TrainingNode()
        child_left.sample_idxs = left_idxs
        child_left.val = self.get_plurality_label(sample_idxs=left_idxs)
        child_left.enropy = self.calc_entropy(sample_idxs=left_idxs)
        child_left.depth = self.nodes[node_idx].depth + 1
        self.nodes[node_idx].child_left = len(self.nodes)
        self.nodes.append(child_left)

        child_right = TrainingNode()
        child_right.sample_idxs = right_idxs
        child_right.val = self.get_plurality_label(sample_idxs=right_idxs)
        child_right.enropy = self.calc_entropy(sample_idxs=right_idxs)
        child_right.depth = self.nodes[node_idx].depth + 1
        self.nodes[node_idx].child_right = len(self.nodes)
        self.nodes.append(child_right)

        self.split_node(node_idx=self.nodes[node_idx].child_left)
        self.split_node(node_idx=self.nodes[node_idx].child_right)
    
    def get_plurality_label(self, sample_idxs: list[int]) -> int:
        counts: dict[int, int] = {}
        for sample_idx in sample_idxs:
            if not self.samples[sample_idx].label in counts:
                counts[self.samples[sample_idx].label] = 0
            counts[self.samples[sample_idx].label] += 1
        max_count = -1
        max_labels = []
        for label in counts:
            if counts[label] >= max_count:
                max_count = counts[label]
                max_labels.append(label)
        return random.sample(max_labels,1)[0]
    
    def calc_info_gain_cat(self, ftr_idx: int, group_idx: int, node_idx: int) -> float:
        left_idxs = []
        right_idxs = []
        for sample_idx in self.nodes[node_idx].sample_idxs:
            if not ftr_idx in self.samples[sample_idx].features:
                continue
            if self.samples[sample_idx].features[ftr_idx] in self.cat_ftr_groupings[ftr_idx][group_idx]:
                left_idxs.append(sample_idx)
            else:
                right_idxs.append(sample_idx)
        if len(left_idxs) < self.min_leaf_size or len(right_idxs) < self.min_leaf_size:
            return -999.9
        s_left = self.calc_entropy(sample_idxs=left_idxs)
        s_right = self.calc_entropy(sample_idxs=right_idxs)
        n_left = len(left_idxs)
        n_right = len(right_idxs)
        info_gain = self.nodes[node_idx].enropy - (n_left*s_left + n_right*s_right)/(n_left + n_right)
        return info_gain

    def calc_info_gain(self, ftr_idx: int, thresh: float, node_idx: int) -> float:
        left_idxs = []
        right_idxs = []
        for sample_idx in self.nodes[node_idx].sample_idxs:
            if not ftr_idx in self.samples[sample_idx].features:
                continue
            if self.samples[sample_idx].features[ftr_idx] < thresh:
                left_idxs.append(sample_idx)
            else:
                right_idxs.append(sample_idx)
        if len(left_idxs) < self.min_leaf_size or len(right_idxs) < self.min_leaf_size:
            return -999.9
        s_left = self.calc_entropy(sample_idxs=left_idxs)
        s_right = self.calc_entropy(sample_idxs=right_idxs)
        n_left = len(left_idxs)
        n_right = len(right_idxs)
        info_gain = self.nodes[node_idx].enropy - (n_left*s_left + n_right*s_right)/(n_left + n_right)
        return info_gain

    def calc_entropy(self, sample_idxs: list[int]) -> float:
        counts: dict[int, int] = {}
        for sample_idx in sample_idxs:
            if not self.samples[sample_idx].label in counts:
                counts[self.samples[sample_idx].label] = 0
            counts[self.samples[sample_idx].label] += 1
        entropy = 0.0
        for label in counts:
            frac = counts[label]/len(sample_idxs)
            entropy -= frac*math.log(frac)
        return entropy
    
    def fit(self) -> None:
        top_node = TrainingNode()
        top_node.sample_idxs = [idx for idx in self.samples]
        top_node.val = self.get_plurality_label(sample_idxs=top_node.sample_idxs)
        top_node.enropy = self.calc_entropy(sample_idxs=top_node.sample_idxs)
        top_node.depth = 0
        self.nodes.append(top_node)
        self.split_node(node_idx=0)

    def oob_fit(self, sample_frac: float) -> dict[int, int]:
        self.nodes = []
        labeled_idxs: dict[int, list[int]] = {}
        for sample_idx in self.samples:
            if not self.samples[sample_idx].label in labeled_idxs:
                labeled_idxs[self.samples[sample_idx].label] = []
            labeled_idxs[self.samples[sample_idx].label].append(sample_idx)
        min_count = min([len(labeled_idxs[label]) for label in labeled_idxs])
        n_to_draw = int(min_count*sample_frac)
        train_idxs = []
        oob_idxs = []
        for label in labeled_idxs:
            for sample_idx in random.sample(labeled_idxs[label], n_to_draw):
                train_idxs.append(sample_idx)
        for sample_idx in self.samples:
            if not sample_idx in train_idxs:
                oob_idxs.append(sample_idx)

        top_node = TrainingNode()
        top_node.sample_idxs = [idx for idx in train_idxs]
        top_node.val = self.get_plurality_label(sample_idxs=top_node.sample_idxs)
        top_node.enropy = self.calc_entropy(sample_idxs=top_node.sample_idxs)
        top_node.depth = 0
        self.nodes.append(top_node)
        self.split_node(node_idx=0)

        oob_preds = {}
        oob_svs = {}
        for sample_idx in oob_idxs:
            oob_preds[sample_idx] = self.predict(self.samples[sample_idx])
            oob_svs[sample_idx] = self.calc_shap_values(self.samples[sample_idx])
        return oob_preds, oob_svs

    def predict(self, sample: cohort.Sample) -> int:
        node_idx = 0
        last_node_idx = 0
        while node_idx >= 0:
            last_node_idx = node_idx
            node_idx = self.nodes[node_idx].predict(sample)
        return self.nodes[last_node_idx].val
    
    def eject_predict(self, sample: cohort.Sample, use_ftr_idxs: list[int]) -> int:
        node_idx = 0
        last_node_idx = 0
        while node_idx >= 0:
            last_node_idx = node_idx
            if not self.nodes[node_idx].ftr_idx in use_ftr_idxs:
                return self.nodes[node_idx].val
            node_idx = self.nodes[node_idx].predict(sample)
        return self.nodes[last_node_idx].val
    
    def calc_shap_values_old(self, sample: cohort.Sample) -> list[float]:
        svs = [0 for _ in range(len(sample.features))]
        ftr_idxs = self.get_ftr_path(sample)
        for iftr_idx in ftr_idxs:
            ftr_idxs = []
            for jftr_idx in ftr_idxs:
                if not iftr_idx == jftr_idx:
                    ftr_idxs.append(jftr_idx)
            for idraw in range(len(ftr_idxs)):
                pre_factor = 1.0/(len(ftr_idxs)*math.comb(len(ftr_idxs) - 1, idraw))
                for combo in itertools.combinations(ftr_idxs, idraw):
                    plus_set = [idx for idx in list(combo)].append(iftr_idx)
                    svs[iftr_idx] += pre_factor*(self.eject_predict(sample, plus_set) - self.eject_predict(sample, list(combo)))
        return svs

    def get_ftr_path(self, sample: cohort.Sample) -> list[int]:
        ftr_idxs = []
        node_idx = 0
        last_node_idx = 0
        while node_idx >= 0:
            ftr_idxs.append(self.nodes[last_node_idx].ftr_idx)
            last_node_idx = node_idx
            node_idx = self.nodes[node_idx].predict(sample)
        return list(set(ftr_idxs))
    
    def calc_shap_values(self, sample: cohort.Sample) -> list[float]:
        svs = [0 for _ in range(len(sample.features))]
        use_idxs = []
        node_vals = []
        node_idx = 0
        last_node_idx = 0
        ftr_counts = {}
        while node_idx >= 0:
            this_ftr = self.nodes[last_node_idx].ftr_idx
            node_vals.append(self.nodes[last_node_idx].val)
            if this_ftr in ftr_counts:
                ftr_counts[this_ftr] += 1
            else:
                ftr_counts[this_ftr] = 1
            use_idxs.append(this_ftr)
            last_node_idx = node_idx
            node_idx = self.nodes[node_idx].predict(sample)
        unique_path = []
        unique_vals = []
        for ii, idx in enumerate(use_idxs):
            if ftr_counts[idx] > 1:
                ftr_counts[idx] = ftr_counts[idx] - 1
            else:
                unique_path.append(idx)
                unique_vals.append(node_vals[ii])
        for ii in range(len(unique_vals)):
            if ii > 0:
                unique_vals[ii] = unique_vals[ii]/(ii*(ii+1))
        for iftr, ftr_idx in enumerate(unique_path):
            svs[ftr_idx] = self.predict(sample)/len(unique_path) - unique_vals[iftr]
            for val in unique_vals[iftr+1:]:
                svs[ftr_idx] += val
        return svs

def draw_ROC_curve(case_scores: list[float] = [], control_scores: list[float] = [], save_path: str = './ROC.png') -> None:
    draw_x = []
    draw_y = []
    vals = case_scores + control_scores
    for thresh in sorted([vv for vv in vals]):
        sens = 0
        for score in case_scores:
            if score > thresh:
                sens += 1
        sens /= len(case_scores)

        spec = 0
        for score in control_scores:
            if score <= thresh:
                spec += 1
        spec /= len(control_scores)

        draw_x.append(1.0-spec)
        draw_y.append(sens)
    
    res = scipy.stats.mannwhitneyu(x=case_scores, y=control_scores)
    cles = res.statistic/(len(case_scores)*len(control_scores))

    # CI for AUC: Hanley and McNeil (1982)
    q1 = cles/(2-cles)
    q2 = 2*cles**2/(1+cles)
    n1 = len(case_scores)
    n2 = len(control_scores)
    se = ((cles*(1-cles) + (n1-1)*(q1-cles**2) + (n2-1)*(q2-cles**2))/(n1*n2))**0.5
    ub = cles + 1.96*se
    lb = cles - 1.96*se

    plt.figure()
    plt.plot(draw_x, draw_y, label='model')
    plt.legend()
    plt.title('AUC: %.3f [%.3f-%.3f], MW-p: %.3e'%(cles, lb, ub, res.pvalue))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()