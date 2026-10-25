## Decision

Rawaj Pay needs to decide which instant transfers should be flagged for investigation as potential mule activity. The fraud team has five analysts, so the threshold should limit the number of flagged cases to a manageable workload. False positives waste analysts' time, while false negatives allow fraudulent transfers to go undetected. The final threshold should reflect these trade-offs.

## Metric

Expected cost is the main decision metric because false positives and false negatives have different costs. We should compare candidate thresholds using the costs of both types of error and select the threshold with the lowest expected cost. Accuracy alone is insufficient because it does not reflect the different consequences of these errors or the analysts' limited capacity.

## Baseline

The baseline is the existing fraud-review process without a model-based risk score. We should compare the proposed approach against this process using the same measures of operational workload and fraud-detection outcomes. The model should demonstrate that its threshold provides a better trade-off between missed mule activity and unnecessary investigations.

## Classical ML or LLM

A classical machine-learning model is the more suitable starting point for scoring structured transfer data. Its scores can be evaluated using historical labels, and its decision threshold can be selected according to the expected costs of false positives and false negatives. An LLM is not necessary for this scoring task; the priority is reliable data, measurable performance, and a threshold aligned with the fraud team's operational needs.
