1. [ AIエージェントのHuman-in-the-Loop評価を深化させる - LayerX エンジニアブログ ]( https://tech.layerx.co.jp/entry/2026/04/01/150000 )
2. Ref: [ AIエージェントのHuman-in-the-Loopを定量評価する #評価指標 - Qiita ]( https://qiita.com/cvusk/items/fe61b526babf45429ba1 )

---

Observe HITL count with OTEL.

## 1. RMSE | Regression

The decrease of HITL is risk and the increase of HITL is a task, but RMSE miss catching direction.

So we have to asymmetric loss function.

## 2. F beta score | Classification

If Recall increase, FN decrease so we have to emphasize Recall.

So we have to use F-beta-score with beta >= 0.

> [!TIPS]
> you think making beta to infinity or zero.


## 3. Timing

Too late checking increases the cost than ealy checking because the effect scope increases if you have to change.

## 4. Type occurance of HITL

1. Clustered with Geni Cofficient
2. Distributed with Variation Coefficient 
3. Front-Loaded
4. Burstiness score

## 5. Dependency

Cascade: 
- Cost of Downstream * Error Probability of Upstream

Multiple HITL competing for human attention:
- N_window * DegrationRate * ReturnDecision

Complementary Synergy
- (T_1 - T_k) * Cost of person in charge

~~~
~~~


