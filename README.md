# Federated Learning Optimizer Comparison Dashboard
### Sakshi Saraswat | Research Internship, NIT Delhi 2025 | GWECA, Ajmer

An interactive Streamlit dashboard visualizing real experimental results comparing **GWO**, **PSO**, and **ABC** optimizers in a decentralized federated learning setup with fog-cloud simulation.

---

## Research Architecture

```
10 Drones (EfficientNet-B0)
       ↓  local training (5 epochs)
  Fog Broker (Router A + Router B)
       ↓  model filtering & fitness scoring
   Cloud VM  →  FedAvg aggregation
       ↓  global model broadcast
10 Drones (next round)
```

- **Model:** EfficientNet-B0 (fine-tuned, 7-class image classification)
- **Federation:** FedAvg over 10 communication rounds
- **Optimizers compared:** GWO (Grey Wolf), PSO (Particle Swarm), ABC (Artificial Bee Colony)
- **Metrics tracked:** Global accuracy, total latency, cloud latency, avg system cost, fog broker fitness (A+B), per-drone loss

---

## Dashboard Charts

| Chart | Insight |
|---|---|
| Global accuracy per round | Which optimizer learns fastest |
| Total latency per round | Communication efficiency |
| Fog broker fitness | How well each optimizer scores models |
| System cost | Computational overhead comparison |
| Acceptance rate | Router A vs B filtering behaviour |
| Overall radar | Normalized multi-metric comparison |
| Per-drone heatmap | Loss across all 10 drones × 10 rounds |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data files in the same folder as dashboard.py
#    gwo_federated_logs.xls
#    fed_pso_effnetb0_logs.xls
#    fed_logs_abc_optimizer.xls

# 3. Run
streamlit run dashboard.py
```

The dashboard auto-loads files from the same directory. You can also upload them manually via the sidebar.

---

## Data File Format

Your files are CSV-formatted `.xls` files with these columns:

```
Round, Global Accuracy (%), Total Latency (ms), Cloud Latency (ms),
Average Cost, Acceptance Rate A / Acceptance A, Acceptance Rate B / Acceptance B,
Avg Fitness A / Fitness A, Avg Fitness B / Fitness B,
Drone 1 Loss, Drone 2 Loss, ..., Drone 10 Loss
```

The dashboard handles minor column name differences between GWO/PSO/ABC files automatically.

---

## Deploy Online (Free) — Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Set **main file** to `dashboard.py`
5. You get a public URL — put this on your resume!

---

## Tech Stack

- Python 3.9+
- Streamlit — dashboard framework
- Plotly — interactive charts
- Pandas & NumPy — data processing
