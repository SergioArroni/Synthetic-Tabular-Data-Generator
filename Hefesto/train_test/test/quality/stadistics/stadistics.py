class Stadistics:
    """Clase base para realizar pruebas estadísticas entre dos conjuntos de datos."""

    def __init__(self, data, path: str):
        self.data = data
        self.path = path

    def execute(self):
        raise NotImplementedError