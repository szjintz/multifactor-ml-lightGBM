class PortfolioConstraints:
    def __init__(self, config: dict):
        self.turnover_limit = config.get("turnover_limit", 0.30)
        self.max_weight = config.get("max_weight", 0.05)
        self.sector_neutral = config.get("sector_neutral", True)
        self.size_neutral = config.get("size_neutral", True)
        self.sector_dev = config.get("sector_dev", 0.02)
        self.cap_dev = config.get("cap_dev", 0.02)
