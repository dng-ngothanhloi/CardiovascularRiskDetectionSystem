# Makefile — CardiovascularRiskDetectionSystem
# Pipeline: EDA → Preprocess → FE → Train → L1 Eval → L2 Interpret → L3 Discovery

.PHONY: help install setup check-config check-data \
	eda preprocess fe fe-rf fe-tabnet \
	train train-rf train-tabnet train-vae-tabnet \
	evaluate interpret attention calibrate calibrate-rf calibrate-tabnet \
	lifestyle-prep xai \
	discover discover-fast \
	all all-fast pipeline-l1 \
	clean clean-models clean-eda clean-fe \
	test lint format docs

# Must run from project root
ifneq ($(wildcard src/),)
ifneq ($(wildcard configs/),)
    # ok
else
    $(warning Warning: configs/ directory not found. Make sure you're in the project root.)
endif
else
    $(error Error: Makefile must be run from project root. Current: $(CURDIR). Use './make.sh' or 'cd' to project root.)
endif

CONFIG        := configs/default.yaml
L3_CONFIG     := configs/l3_discovery.yaml
PYTHON        := python3
SRC           := src
DATA          := data/cardio_data.csv

ARTIFACTS     := artifacts
EDA_DIR       := $(ARTIFACTS)/eda
PRE_DIR       := $(ARTIFACTS)/preprocessing
FE_DIR        := $(ARTIFACTS)/fe
MODEL_DIR     := $(ARTIFACTS)/Model
EVAL_DIR      := $(ARTIFACTS)/Eval
INTERPRET_DIR := $(ARTIFACTS)/Interpretation
DISCOVERY_DIR := $(ARTIFACTS)/Discovery
REPORTS       := reports

N_BOOT        := 2000
SEED          := 42

help: ## Show available targets
	@echo "CardiovascularRiskDetectionSystem — Make targets"
	@echo ""
	@echo "Setup:"
	@echo "  make setup | install | check-data | check-config"
	@echo ""
	@echo "Pipeline stages (in order):"
	@echo "  make eda              # EDA → artifacts/eda/"
	@echo "  make preprocess       # Clean data → artifacts/preprocessing/"
	@echo "  make fe               # FE for rf + tabnet → artifacts/fe/"
	@echo "  make train            # Train RF + TabNet (requires --model each)"
	@echo "  make train-rf | train-tabnet | train-vae-tabnet"
	@echo "  make evaluate         # L1 OOF + Test metrics → artifacts/Eval/"
	@echo "  make interpret        # L2 attention + calibrate + lifestyle-prep + XAI"
	@echo "  make discover         # L3 Apriori discovery → artifacts/Discovery/"
	@echo ""
	@echo "Full runs:"
	@echo "  make pipeline-l1      # eda → preprocess → fe → train → evaluate"
	@echo "  make all              # pipeline-l1 → interpret → discover"
	@echo "  make all-fast         # same with CONFIG=configs/fast.yaml"
	@echo ""
	@echo "Config override:"
	@echo "  make train-rf CONFIG=configs/no_hpo.yaml"
	@echo "  make evaluate N_BOOT=200"
	@echo ""
	@echo "Detail: docs/Flow-Pipeline.md"

install: ## Install Python dependencies
	pip install -r requirements.txt
	@echo "Installation complete."

check-config:
	@if [ ! -f $(CONFIG) ]; then echo "Error: $(CONFIG) not found!"; exit 1; fi
	@echo "Config OK: $(CONFIG)"

check-data:
	@if [ ! -f $(DATA) ]; then echo "Error: $(DATA) not found!"; exit 1; fi
	@echo "Data OK: $(DATA)"

setup: install check-config check-data ## Install deps and verify files
	@echo "Setup complete."

# ---------- Stage 1–3: EDA / Preprocess / FE ----------

eda: ## Exploratory data analysis → artifacts/eda/
	@mkdir -p $(EDA_DIR)
	$(PYTHON) -m $(SRC).data_understanding --config $(CONFIG)
	@echo "EDA → $(EDA_DIR)/"

preprocess: ## Preprocessing → artifacts/preprocessing/
	@mkdir -p $(PRE_DIR)
	$(PYTHON) -m $(SRC).preprocessing --config $(CONFIG)
	@echo "Preprocess → $(PRE_DIR)/"

fe-rf: ## Feature engineering for RandomForest
	@mkdir -p $(FE_DIR)/random_forest
	$(PYTHON) -m $(SRC).feature_engineering --config $(CONFIG) --model-type rf
	@echo "FE RF → $(FE_DIR)/random_forest/"

fe-tabnet: ## Feature engineering for TabNet
	@mkdir -p $(FE_DIR)/tabnet
	$(PYTHON) -m $(SRC).feature_engineering --config $(CONFIG) --model-type tabnet
	@echo "FE TabNet → $(FE_DIR)/tabnet/"

fe: fe-rf fe-tabnet ## Feature engineering for both model families
	@echo "FE complete (rf + tabnet)."

# ---------- Stage 4: Train (HPO runs inside modeling) ----------

train-rf: ## Train RandomForest (5-fold CV + optional HPO/test scoring)
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) -m $(SRC).modeling --config $(CONFIG) --model rf
	@echo "RF → $(MODEL_DIR)/rf/"

train-tabnet: ## Train TabNet
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) -m $(SRC).modeling --config $(CONFIG) --model tabnet
	@echo "TabNet → $(MODEL_DIR)/tabnet/"

train-vae-tabnet: ## Train VAE-TabNet (joint training)
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) -m $(SRC).modeling --config $(CONFIG) --model vae_tabnet --joint
	@echo "VAE-TabNet → $(MODEL_DIR)/vae_tabnet/"

train: train-rf train-tabnet ## Train RF + TabNet (HPO is inside each modeling run)
	@echo "Training RF + TabNet complete."

# ---------- Stage 5: L1 Evaluation ----------

evaluate: ## L1 evaluation (OOF + held-out Test if present) → artifacts/Eval/
	@mkdir -p $(EVAL_DIR)
	$(PYTHON) -m $(SRC).evaluation \
		--model_root $(MODEL_DIR) \
		--out_dir $(EVAL_DIR) \
		--n_boot $(N_BOOT) \
		--seed $(SEED)
	@echo "Evaluation → $(EVAL_DIR)/ (report.md, summary.json)"

# ---------- Stage 6: L2 Interpretation ----------

attention: ## TabNet attention + RF importance comparison
	@mkdir -p $(INTERPRET_DIR)
	$(PYTHON) -m $(SRC).interpretation.run_attention_summary \
		--fe_dir_tabnet $(FE_DIR)/tabnet \
		--model_root $(MODEL_DIR)/tabnet \
		--out_root $(INTERPRET_DIR) \
		--rf_model_root $(MODEL_DIR)/rf \
		--fe_dir_rf $(FE_DIR)/random_forest
	@echo "Attention → $(INTERPRET_DIR)/"

calibrate-rf: ## L2 calibration for RF (Platt on OOF)
	@mkdir -p $(INTERPRET_DIR)/calibration
	$(PYTHON) -m $(SRC).interpretation.run_calibration \
		--model rf --method platt \
		--model_root $(MODEL_DIR) \
		--out_root $(INTERPRET_DIR)/calibration \
		--n_bins 15

calibrate-tabnet: ## L2 calibration for TabNet (Isotonic on OOF)
	@mkdir -p $(INTERPRET_DIR)/calibration
	$(PYTHON) -m $(SRC).interpretation.run_calibration \
		--model tabnet --method isotonic \
		--model_root $(MODEL_DIR) \
		--out_root $(INTERPRET_DIR)/calibration \
		--n_bins 15

calibrate: calibrate-rf calibrate-tabnet ## L2 calibration for RF + TabNet

lifestyle-prep: ## Test spotlight + lifestyle join (required before xai lifestyle)
	@mkdir -p $(INTERPRET_DIR)/test_prediction_result
	$(PYTHON) -m $(SRC).test_prediction_verification \
		--n_samples 0 \
		--test_prediction src/test_prediction_input.txt
	$(PYTHON) -m $(SRC).create_lifestyle_enriched_predictions \
		--fe_dir $(FE_DIR) \
		--predictions_dir $(INTERPRET_DIR)/test_prediction_result \
		--output_dir $(INTERPRET_DIR)/test_prediction_result

xai: lifestyle-prep ## Lifestyle / XAI charts under Interpretation/reports
	@mkdir -p $(INTERPRET_DIR)/reports
	$(PYTHON) -m $(SRC).xai_interpretability --output-dir $(INTERPRET_DIR)/reports

interpret: attention calibrate xai ## Full L2: attention + calibration + lifestyle-prep + XAI
	@echo "L2 Interpretation → $(INTERPRET_DIR)/"

# ---------- Stage 7: L3 Discovery ----------

discover: ## L3 TabNet-attention → Apriori rules
	@mkdir -p $(DISCOVERY_DIR)
	$(PYTHON) -m $(SRC).run_discovery --config $(L3_CONFIG)
	@echo "Discovery → $(DISCOVERY_DIR)/"

discover-fast: ## L3 with faster settings (no bootstrap, max_len=3)
	@mkdir -p $(DISCOVERY_DIR)
	$(PYTHON) -m $(SRC).run_discovery --config $(L3_CONFIG) \
		--tau 0.5 --min_support 0.02 --min_conf 0.6 --min_lift 1.2 \
		--max_len 3 --no_bootstrap
	@echo "Discovery (fast) → $(DISCOVERY_DIR)/"

# ---------- Composed pipelines ----------

pipeline-l1: eda preprocess fe train evaluate ## EDA through L1 evaluation
	@echo ""
	@echo "L1 pipeline done. See $(EVAL_DIR)/report.md"

all: pipeline-l1 interpret discover ## Full pipeline through L3
	@echo ""
	@echo "=========================================="
	@echo "Full pipeline completed"
	@echo "=========================================="
	@echo "  EDA:            $(EDA_DIR)/"
	@echo "  Preprocess:     $(PRE_DIR)/"
	@echo "  FE:             $(FE_DIR)/"
	@echo "  Models:         $(MODEL_DIR)/"
	@echo "  L1 Eval:        $(EVAL_DIR)/"
	@echo "  L2 Interpret:   $(INTERPRET_DIR)/"
	@echo "  L3 Discovery:   $(DISCOVERY_DIR)/"

all-fast: ## Full pipeline using configs/fast.yaml
	$(MAKE) all CONFIG=configs/fast.yaml

# ---------- Maintenance ----------

clean:
	rm -rf $(ARTIFACTS)/*
	rm -rf $(REPORTS)/*
	@echo "Cleaned artifacts/ and reports/"

clean-models:
	rm -rf $(MODEL_DIR)/*
	@echo "Cleaned $(MODEL_DIR)/"

clean-eda:
	rm -rf $(EDA_DIR)/*
	@echo "Cleaned $(EDA_DIR)/"

clean-fe:
	rm -rf $(FE_DIR)/*
	@echo "Cleaned $(FE_DIR)/"

test:
	$(PYTHON) -m pytest tests/ -q || echo "pytest not installed or tests failed"

lint:
	@if command -v flake8 > /dev/null; then \
		flake8 $(SRC) --max-line-length=120 --ignore=E501,W503; \
	else \
		echo "flake8 not installed: pip install flake8"; \
	fi

format:
	@if command -v black > /dev/null; then \
		black $(SRC) --line-length=120; \
	else \
		echo "black not installed: pip install black"; \
	fi

docs:
	@echo "Design docs: docs/Flow-Pipeline.md docs/L1_Prediction_Summary.md docs/L2-L3_DetailDesign.md"
