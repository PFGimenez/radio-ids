from multimodels import *
import numpy as np

class Evaluator:

    """
        Recall (= true positive rate): attaques correctement trouvées / toutes les vraies attaques. Proportion d'attaques qui ont été détectées (si on prédit toujours une attaque, le recall vaut 1)
        Precision: nb attaques correctement trouvées / nb attaques détectées. Proportion de détection pertinentes (si on prédit toujours une attaque, la précision sera proche de 0)
        F-score = 2 * (precision * rappel) / (precision + rappel)
    """

    def __init__(self, identifier, minimum_overlap, all_attack):
        """
            format all_attack : shape (-1,3)
            column 0 : attack start timestamp
            column 1 : attack end itemstamp
            column 2 : attack identifier
        """
        self._id = identifier
        self._minimum_overlap = minimum_overlap
        # TODO : ou remplacer identifier par une fonction (pour agréger toutes les attaques bluetooth par exemple)
        self._attack = all_attack[all_attack[:,2] == identifier]
        print(self._attack.shape[0],"attacks on",identifier)

    def is_in_attack(self, timestamp):
        for a in self._attack:
            if timestamp >= a[0] and timestamp <= a[1]:
                return True
        return False

    def evaluate(self, detected_positive, total_predictions_number):
        """
            Prediction : shape (-1,2)
            column 0 : timestamp
            column 1 : true iff detection
        """
        false_negative = 0
        true_negative = 0
        false_positive = 0
        true_positive = 0

        total_positives = len(detected_positive)
        print("total pos",total_positives)
        true_positive = sum(list(map(self.is_in_attack, detected_positive)))
        false_negative = self._attack.shape[0] - true_positive
        true_negative = total_positives - false_positive
        print(true_positive, true_negative, false_positive, false_negative)

attack = np.loadtxt("/data/data/00.raw/log_attack/Last_Exp_attacks/logattack")
print(attack.shape)
att1 = Evaluator(2440, 0, attack)
print(att1._attack)
example = np.random.randint(500, 300000, (1000, 1))
#print(example)
att1.evaluate(example, 4000)
exit()

nb_features = 2944

files = get_files_names(["/data/data/00.raw/raw/Adr_Expe_28-08_07-10/raspi1/"], "01_October")
files = ["features-"+d.split("/")[-1] for d in files]
print(files)
data = np.concatenate([np.fromfile(f).reshape(-1, nb_features + 1) for f in files])

print(data.shape)

#models = MultiModels()
#models.load("micro-ocsvm.joblib")



