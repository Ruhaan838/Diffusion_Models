class EMA: #exponatial moving average
    """
    w = ß * w_old + (1 - ß) * w_new
    ß = 0.99
    """
    def __init__(self, beta):
        self.beta = beta
        self.step_num = 0
        
    def update_model_(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weigths, new_weigths = ema_param.data, current_param.data
            ema_param.data = self.update_avg(old_weigths, new_weigths)
        
    def update_avg(self, old, new):
        return self.beta * old + (1 - self.beta) * new
    
    def step(self, ema_model, model, step_start_ema=2000):
        if self.step_num < step_start_ema:
            self.reset(ema_model, model)
            self.step_num += 1
            return
        self.update_model_(ema_model, model)
        self.step_num += 1
            
    def reset(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
