# Workshop Results

## Setup

| Instance | Model | Query strategy | Labeler | Labels | Iters |
|---|---|---|---|---:|---:|
| 6 | svm | uncertainty sampling least confidence | ddardanis | 3 | 3 |
| 7 | logistic regression | uncertainty sampling least confidence | alberto.marchinelli@gmail.com | 8 | 8 |
| 8 | random forest | Workshop | roberto.costa@gft.com | 4 | 5 |
| 9 | random forest | CLUE | fise@gft.com | 6 | 7 |

## Metrics (on added labels)

| Instance | Avg latency (ms) | Model/Labeler | Model/GT | Labeler/GT |
|---|---:|---:|---:|---:|
| 6 | 47,250 (47.3 s) | 66.67% (2/3) | 66.67% (2/3) | 33.33% (1/3) |
| 7 | 80,025 (80.0 s) | 50.00% (4/8) | 12.50% (1/8) | 25.00% (2/8) |
| 8 | 139,069 (139.1 s) | 50.00% (2/4) | 0.00% (0/4) | 0.00% (0/4) |
| 9 | 71,416 (71.4 s) | 100.00% (6/6) | 66.67% (4/6) | 66.67% (4/6) |

- Avg latency: avg time that labeler spent on the ticket
- Model/Labler: percentage of agreement between the model and the labeler
- Model/GT: percentage of agreement between the model and the actual (ground truth) ticket labels
- labeler/GT: percentage of agreement between the labeler and the actual ticket labels



## Held-out Macro F1 per iteration

| Instance | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 | iter 6 | iter 7 | iter 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.0067 | 0.0044 | 0.0389 | — | — | — | — | — |
| 7 | 0.4540 | 0.4906 | 0.5502 | 0.5426 | 0.5514 | 0.5537 | 0.5528 | 0.5616 |
| 8 | 0.2603 | 0.2406 | 0.2927 | 0.2861 | 0.3132 | — | — | — |
| 9 | 0.2603 | 0.2613 | 0.2601 | 0.2311 | 0.2129 | 0.2554 | 0.2815 | — |
