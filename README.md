# ML_project


This project uses machine learning models to predict NBA 2K26 overall player ratings based on advanced basketball analytics. By leveraging metrics such as Player Efficiency Rating (PER), Box Plus-Minus (BPM), and Value Over Replacement Player (VORP), the project explores how closely NBA 2K player ratings align with real-world performance data.
Three regression models are evaluated:
Linear Regression
Random Forest Regressor
XGBoost Regressor
The results demonstrate that nonlinear models, particularly XGBoost, are highly effective at approximating NBA 2K ratings.

# NBA 2K26 Jumpshot Meta Detector

**ML-powered anomaly detection system** that identifies overpowered jumpshots and stealth patch nerfs/buffs by analyzing real player test data.

## What it does
- Trains XGBoost baseline model on jumpshot configs (base/release/speed/3PT rating)
- Computes residuals: `actual make% - expected make%`
- Ranks "meta" jumpshots by advantage (positive residuals)
- Detects patch impacts across configs

## Real Results (25 sessions, 2,500 shots)
 
Est. greens calculated (92 3PT → ~75% of makes)
Loaded 25 shot sessions
Baseline model trained + saved
Train R2: 0.9989
Test  R2: 0.2902

EXPLOIT LEADERBOARD (top 5)
       base  release1 release2 speed  avg_residual  exploit_score  total_attempts  total_greens
4     Curry      Kobe      Gay   max      0.021233       0.159246             200            91
7   Iverson      Bird      Gay  full      0.031998       0.147674             100            43
8    Jordan      Kobe      Gay   3/4      0.015203       0.114021             200            85
5     Curry  RayAllen     Kobe  full      0.003656       0.027423             200            89
11       KD  ZachEdey  Unknown   max      0.002142       0.009884             100            45

