class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = DotDict()


config.state_space_limits = [[-4.8, 4.8], [-4, 4], [-0.418, 0.418], [-4, 4]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]

config.qt_size_array = [11, 11, 11, 11] # [Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
config.reward_values = {0 : -10, 1 : 1}
config.discount_factor = 0.95
config.learning_rate = 0.005
config.epsilon = 0.9
config.epsilon_decay = 0.9999
config.episodes = 100000
config.graph = 1
config.render = 1