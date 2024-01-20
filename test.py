import csv
import cohort
import tree
import numpy as np

class NHANESI_cohort(cohort.Cohort):
    def __init__(self) -> None:
        super().__init__()
        with open('./NHANESI_is_categorical.csv', 'r') as infile:
            reader = csv.reader(infile)
            next(reader)
            ftr_idx = 1
            for line in reader:
                self.feature_names[ftr_idx] = line[0]
                if int(line[1]) == 1:
                    self.is_ftr_categorical[ftr_idx] = True
                else:
                    self.is_ftr_categorical[ftr_idx] = False
    
    def read_csv(self, fpath: str) -> None:
        with open(fpath, 'r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            ftr_idxs = {}
            headers = next(reader)
            id_idx = headers.index('SampleID')
            label_idx = headers.index('death')
            for ftr_idx in self.feature_names:
                ftr_idxs[ftr_idx] = headers.index(self.feature_names[ftr_idx])
            for line in reader:
                sample = cohort.Sample()
                sample.label = int(line[label_idx])
                for ftr_idx in self.is_ftr_categorical:
                    if self.is_ftr_categorical[ftr_idx]:
                        sample.features[ftr_idx] = int(line[ftr_idxs[ftr_idx]])
                    else:
                        sample.features[ftr_idx] = float(line[ftr_idxs[ftr_idx]])
                self.samples[int(line[id_idx])] = sample

if __name__ == '__main__':
    training_cohort = NHANESI_cohort()
    training_cohort.read_csv('./NHANESI_training.csv')
    training_cohort.calc_categorical_feature_groupings()
    training_cohort.calc_feature_thresholds()
    this_tree = tree.Tree(training_cohort=training_cohort)
    oob_preds: dict[int, list[int]] = {sample_idx: [] for sample_idx in training_cohort.samples}
    for ii in range(1000):
        print(ii)
        these_preds = this_tree.oob_fit(sample_frac=0.667)
        for sample_idx in these_preds:
            oob_preds[sample_idx].append(these_preds[sample_idx])
    oob_scores: dict[int, float] = {sample_idx: np.mean(oob_preds[sample_idx]) for sample_idx in oob_preds}
    case_scores = []
    control_scores = []
    for sample_idx in oob_scores:
        if training_cohort.samples[sample_idx].label == 1:
            case_scores.append(oob_scores[sample_idx])
        else:
            control_scores.append(oob_scores[sample_idx])
    tree.draw_ROC_curve(case_scores=case_scores, control_scores=control_scores)