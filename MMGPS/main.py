import simpy
from gene import Gene


if __name__ == "__main__":
    lambd = 0.1  # rate from inactive -> active
    mu =    0.2  # rate from active -> inactive
    nu =    0.3  # rate from active -> product
    delta = 0.4  # rate from product -> degraded

    # start the environment
    env = simpy.Environment()

    # make a gene
    gene = Gene(env, lambd, mu, nu, delta)

    # run for fixed amount of time
    env.run(until=1500)

    products = [product for product in gene.products if not product.degraded]
    print(f"There are {len(products)} gene products after {env.now} (unitless) time")
