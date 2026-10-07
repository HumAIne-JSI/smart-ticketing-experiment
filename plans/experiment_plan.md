## Experiment Plan

### Overview

**Goal:** Measure whether AI assistance improves ticket-labeling accuracy, speed, and fatigue compared with unaided labeling. Accuracy is evaluated against the existing ground-truth labels.

The study includes **10–15 participants**. Each participant completes one 90-minute Treatment session and one 90-minute Control session on separate days. Participants label as many tickets as they can during each session.

Treatment-first and Control-first assignments are balanced as evenly as possible across participants to counterbalance order effects.

### Ticket Assignment

The pool of approximately 1,000 tickets is randomly split once into two parts of roughly equal size. To prevent a participant from labeling the same ticket in both conditions:

- Half of the participants use split 1 for Treatment and split 2 for Control.
- The other half use split 2 for Treatment and split 1 for Control.

Each condition has its own active-learning instance, model state, and `historical_neighbors` pool. Tickets are selected within the assigned split, and the model retrains after every submitted label.

### Conditions

Control shows the raw ticket and label dropdown. Treatment additionally shows the model prediction and confidence, LIME explanation, and similar tickets (`predicted_class_neighbors` and `historical_neighbors`).

The active-learning loop is identical in both conditions; only the labeler's view differs.

### Data Captured

For each label, capture:

- `participant_id`, `condition`, `ticket_id`, and session/split assignment
- The participant's `label` and the ground-truth label
- `start_time` and `end_time`
- `model_prediction` and model confidence, where available
- `second_model_prediction` and model confidence, where available
- `is_tired`, `is_difficult`, and `i_dont_know`

### Analysis

- Compare Treatment and Control on accuracy, time-to-label, and fatigue indicators, accounting for session order and ticket split.
- Examine changes in speed, accuracy, and fatigue throughout each 90-minute session.
- Analyze the model's learning curve and labeler-acceptance rate over each session.
- Evaluate XAI neighbor quality using neighbor-label agreement, agreement with the participant's label, and LIME-feature overlap.
