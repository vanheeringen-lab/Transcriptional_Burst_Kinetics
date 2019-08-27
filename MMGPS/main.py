import simpy
from gene import Gene
from moment_inference import moment_inference
import numpy as np

if __name__ == "__main__":
    lambd = 2  # rate from inactive -> active
    mu = 0.5  # rate from active -> inactive
    nu = 4  # rate from active -> product
    delta = 1  # rate from product -> degraded

    products = []
    for i in range(150):
        # start the environment
        env = simpy.Environment()

        # make a gene
        gene = Gene(env, lambd, mu, nu, delta)

        # run for fixed amount of time
        env.run(until=3000)

        products.append(len([product for product in gene.products if not product.degraded]))

    print(f'The parameters are: lambda {lambd}, mu {mu}, nu {nu}, delta {delta}')
    print(f'the parameters based on moment inference is: {moment_inference(np.array(products))}')
    print(f"The average of gene-products after {env.now} gene-product half-times is {np.mean(products)},")
    print(f"And theoretically we expect: {((lambd * nu) / ((lambd + mu) * delta))}")
