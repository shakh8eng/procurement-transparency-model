# Transparency, Entry and Prices in Public Procurement

Simulation code for the paper "Transparency, Entry and Prices in Public Procurement: A Simulated Auction Model for Central Asia" (Shohrukh Tojidinov, ECON8001 Independent Research Paper, Marmara University).

## What the code does

- checks the model formulas with a Monte Carlo simulation of 200,000 random tenders for each number of bidders;
- computes the number of bidders and the expected price when transparency changes from 0 to 1;
- calibrates the model to the average number of bidders (currently Kazakhstan's national tenders, 2021 and 2024);
- draws Figure 1 of the paper.

## How to run

```
pip install numpy matplotlib
python simulation.py
```

The script prints the tables used in Section 5 and saves `figure1.pdf`, `figure1.png` and `results_tau.csv`.
