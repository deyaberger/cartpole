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

infos = DotDict()


infos.state_space_limits = [[-4.8, 4.8], [-4, 4], [-0.418, 0.418], [-4, 4]] #[Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]

infos.qt_size_array = [7, 7, 7, 7] # [Cart Pos, Cart Velocity, Pole Angle, Pole Angular Velocity]
infos.reward_values = {0 : -10, 1 : 1}
infos.discount_factor = 0.95
infos.learning_rate = 0.005
infos.epsilon = 0.9
infos.epsilon_decay = 0.9999
infos.episodes = 1000000
infos.graph = 0
infos.render = 1
infos.graph_frequency = 10000
infos.average = 100