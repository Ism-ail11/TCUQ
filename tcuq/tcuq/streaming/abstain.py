class BudgetController:
    """Simple rate controller to respect an abstention budget b (0..1)."""
    def __init__(self, budget=0.1):
        self.budget = budget
        self.n = 0
        self.a = 0

    def allow(self):
        if self.n == 0:
            return True
        return (self.a / self.n) < self.budget

    def update(self, abstained: bool):
        self.n += 1
        if abstained: self.a += 1
