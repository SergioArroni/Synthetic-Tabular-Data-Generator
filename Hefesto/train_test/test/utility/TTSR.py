

class TTSR:
    def __init__(self, model, device, test_loader, save_path):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.save_path = save_path