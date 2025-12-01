import pickle
with open('fleet_info_副本.pkl', 'rb') as f:
    loaded_fleet_info = pickle.load(f)
    print('fleet_info:', loaded_fleet_info)
