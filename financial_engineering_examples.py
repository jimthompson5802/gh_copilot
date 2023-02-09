# function to calculate Black-Scholes option price
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


# function to compute the Greeks
def greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return delta, gamma, vega, theta, rho


# function to generate 300 interest rate paths with a two-factor model
def generate_paths(S0, r0, sigma, rho, T, N, M):
    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    for t in range(1, N + 1):
        z1 = np.random.standard_normal(M)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.standard_normal(M)
        paths[:, t] = paths[:, t - 1] * np.exp((r0 - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z1)
    return paths


# function to compute the option price via Monte Carlo simulation
def monte_carlo(S0, K, T, r, sigma, option_type, N, M):
    dt = T / N
    paths = generate_paths(S0, r, sigma, 0.0, T, N, M)
    payoff = np.maximum(0, paths[:, -1] - K) if option_type == 'call' else np.maximum(0, K - paths[:, -1])
    price = np.exp(-r * T) * np.sum(payoff) / M
    return price
