import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)

    # Features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.normal(50000, 15000, n_samples).clip(10000, 200000)
    debts = np.random.normal(20000, 10000, n_samples).clip(0, 100000)
    payment_history = np.random.uniform(0, 1, n_samples)  # 0: bad, 1: good
    employment_status = np.random.choice(['employed', 'unemployed', 'self-employed'], n_samples)
    credit_score = np.random.normal(600, 100, n_samples).clip(300, 850)

    # Target: creditworthy based on some logic
    creditworthy = (
        (income > 40000) &
        (debts < 30000) &
        (payment_history > 0.7) &
        (credit_score > 650)
    ).astype(int)

    # Add some noise
    noise = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    creditworthy = np.where(noise, 1 - creditworthy, creditworthy)

    data = pd.DataFrame({
        'age': age,
        'income': income,
        'debts': debts,
        'payment_history': payment_history,
        'employment_status': employment_status,
        'credit_score': credit_score,
        'creditworthy': creditworthy
    })

    return data

if __name__ == "__main__":
    data = generate_synthetic_data()
    data.to_csv('credit_data.csv', index=False)
    print("Synthetic dataset generated and saved to credit_data.csv")
