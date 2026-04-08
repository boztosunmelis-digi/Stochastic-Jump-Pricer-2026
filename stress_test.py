"""Stress test scenarios for the Bates stochastic volatility
jump-diffusion model."""
import matplotlib.pyplot as plt
from engine import BatesModelEngine


def run_stress_test():
    """Run and plot standard vs crash stress test scenarios."""

    # Scenario A: Standard Market (Low Jump Risk)
    # Each scenario gets its own engine instance so parameter changes
    # are isolated and do not leak between runs.
    engine_a = BatesModelEngine()
    engine_a.lamb = 0.05
    engine_a.mu_j = -0.02
    path_standard = engine_a.simulate_path()

    # Scenario B: 2026 "Market Crash" (High Jump Risk)
    # Spike Lambda and set a deep negative Mu_j
    engine_b = BatesModelEngine()
    engine_b.lamb = 5.0    # 5 jumps expected per year
    engine_b.mu_j = -0.15  # -15% average jump size
    path_crash = engine_b.simulate_path()

    # Plotting the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(
        path_standard, label="Standard Market (Heston-like)", color='blue'
    )
    plt.plot(path_crash, label="Market Crash (Bates Jumps)", color='red')
    plt.title("Bates Model: 2026-2027 Stress Test Scenario")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_stress_test()
