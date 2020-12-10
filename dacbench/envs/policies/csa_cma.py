def csa(env, state):
    u = env.es.sigma
    hsig = env.es.adapt_sigma.hsig(env.es)
    env.es.hsig = hsig
    delta = env.es.adapt_sigma.update2(env.es, function_values=env.cur_obj_val)
    u *= delta
    return u
