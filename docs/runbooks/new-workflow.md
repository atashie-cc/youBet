# How to Add a New Prediction Domain

1. **Copy the template**:
   ```bash
   cp -r workflows/_template workflows/my_new_domain
   ```

2. **Edit `CLAUDE.md`**: Add domain-specific context, data sources, feature descriptions.

3. **Edit `config.yaml`**: Define features, model parameters, data source URLs, and split configuration.

4. **Implement scripts**:
   - `collect_data.py` — fetch/download raw data
   - `build_features.py` — transform to feature matrix with differentials
   - `train.py` — train model with calibration
   - `predict.py` — generate predictions
   - `backtest.py` — evaluate on historical data

5. **Start the research log**: Create first entry in `research/log.md`.

6. **Add tests**: At minimum, test data collection and feature building.

7. **Update root `workflows/README.md`** with a link to the new workflow.
