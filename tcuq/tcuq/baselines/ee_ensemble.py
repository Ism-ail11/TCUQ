# Scaffold; real EE branches not required for core TCUQ repro.
class EarlyExitEnsemble:
    def __init__(self, backbone):
        self.backbone = backbone
    def predict(self, x):
        return self.backbone(x)  # placeholder
