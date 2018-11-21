from multimodels import *
import numpy as np

class Evaluator:

    """
        Recall (= true positive rate): attaques correctement trouvées / toutes les vraies attaques. Proportion d'attaques qui ont été détectées (si on prédit toujours une attaque, le recall vaut 1)
        Precision: nb attaques correctement trouvées / nb attaques détectées. Proportion de détection pertinentes (si on prédit toujours une attaque, la précision sera proche de 0)
        F-score = 2 * (precision * rappel) / (precision + rappel)
    """

    def __init__(self, identifier, all_attack):
        """
            format all_attack : shape (-1,3)
            column 0 : attack start timestamp
            column 1 : attack end itemstamp
            column 2 : attack identifier
        """
        self._id = identifier
        # TODO : ou remplacer identifier par une fonction (pour agréger toutes les attaques bluetooth par exemple)
        self._attack = all_attack[all_attack[:,2] == identifier]
        print(self._attack.shape[0],"attacks on",identifier)

    def is_in_attack(self, timestamp):
        for a in self._attack:
            if timestamp >= a[0] and timestamp <= a[1]:
                return True
        return False

    def evaluate(self, detected_positive, detected_negative):
        """
            Prediction : shape (-1,2)
            column 0 : timestamp
            column 1 : true iff detection
        """

        total_positives = len(detected_positive)
        true_positive = sum(list(map(self.is_in_attack, detected_positive)))
        false_positive = total_positives - true_positive

        total_negatives = len(detected_negative)
        false_negative = sum(list(map(self.is_in_attack, detected_negative)))
        true_negative = total_negatives - false_negative

        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / total_positives

        print("total pos",total_positives, "total negative",total_negatives)
        print("tp",true_positive, "tn",true_negative, "fp",false_positive,"fn", false_negative)
        print("precision",precision,"recall",recall)



attack = np.loadtxt("/data/data/00.raw/log_attack/Last_Exp_attacks/logattack")
print(attack.shape)
att1 = Evaluator(2440, attack)
print(att1._attack)
example_pos = np.random.randint(500, 300000, (100, 1))
example_neg = np.random.randint(500, 300000, (900, 1))
#print(example)
att1.evaluate(example_pos, example_neg)
exit()

nb_features = 2944

files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
files = ["features-"+d.split("/")[-1] for d in files]
print(files)
data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])

print(data.shape)

#models = MultiModels()
#models.load("micro-ocsvm.joblib")



