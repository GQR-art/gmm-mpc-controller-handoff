from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
@dataclass
class JointGMM:
 scaler: StandardScaler
 gmm: GaussianMixture
 input_dim: int
 def save(self,path): joblib.dump(self,path)
 @staticmethod
 def load(path): return joblib.load(path)
