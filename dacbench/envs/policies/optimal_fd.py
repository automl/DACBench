import json


def get_optimum(env, state):
    instance = env.get_instance()[:-12] + "optimal.json"
    with open(instance, "r+") as fp:
        optimal = json.load(fp)
    return optimal[env.c_step]
