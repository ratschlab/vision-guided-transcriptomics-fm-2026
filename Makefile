# Thin wrappers over run.py. Override CONFIG / RUN / SET / ENV on the command line:
#   make all   CONFIG=configs/scgpt.yaml RUN=scgpt
#   make train SET="models.pca_components=50 models.ae.latent_dim=50"
#   make all   ENV=cluster          # site profile from environments.yaml
CONFIG ?= configs/default.yaml
RUN    ?=
SET    ?=
ENV    ?=
PY     ?= python
# Where the per-substrate run directories live. Empty lets scripts/paper_tables.py
# resolve it from the same site profile every stage uses.
ARTIFACTS ?=
TABLES    ?= artifacts/paper_tables

# Only override run_name when RUN is given, so each config keeps its own default.
_SET = $(if $(RUN),--set run_name=$(RUN) $(SET),$(if $(SET),--set $(SET),))
# Empty ENV lets run.py fall back to $VGTFM_ENV, then environments.yaml's default.
_ENV = $(if $(ENV),--env $(ENV),)

.PHONY: all data train eval diagnose integrate results ablate biosignal figures \
        embed paper-tables smoke test embed-check lint format clean help

help:
	@echo "stages:  data train eval diagnose figures        (make all)"
	@echo "opt-in:  embed integrate ablate biosignal results"
	@echo "paper:   paper-tables (cross-run tables; checks the stages agree first)"
	@echo "checks:  test (unit)   smoke (pipeline on a subset)"
	@echo "         embed-check (gene FMs against real checkpoints, one GPU)"
	@echo "style:   lint (check)  format (apply)"
	@echo "vars:    CONFIG=$(CONFIG)  RUN=$(RUN)  SET=$(SET)  ENV=$(ENV)"

all:       ; $(PY) run.py all       --config $(CONFIG) $(_ENV) $(_SET)
data:      ; $(PY) run.py data      --config $(CONFIG) $(_ENV) $(_SET)
train:     ; $(PY) run.py train     --config $(CONFIG) $(_ENV) $(_SET)
eval:      ; $(PY) run.py eval      --config $(CONFIG) $(_ENV) $(_SET)
diagnose:  ; $(PY) run.py diagnose  --config $(CONFIG) $(_ENV) $(_SET)
integrate: ; $(PY) run.py integrate --config $(CONFIG) $(_ENV) $(_SET)
ablate:    ; $(PY) run.py ablate    --config $(CONFIG) $(_ENV) $(_SET)
biosignal: ; $(PY) run.py biosignal --config $(CONFIG) $(_ENV) $(_SET)
figures:   ; $(PY) run.py figures   --config $(CONFIG) $(_ENV) $(_SET)
embed:     ; $(PY) run.py embed     --config $(CONFIG) $(_ENV) $(_SET)
results:   ; $(PY) run.py results   --config $(CONFIG) $(_ENV) $(_SET)

# The manuscript's cross-run tables. Regenerate these whenever any run's `integrate`,
# `eval` or `ablate` has been rerun — they are the one output no stage rebuilds, and
# they drifted a whole batch behind their own source once already.
paper-tables:
	$(PY) scripts/paper_tables.py $(if $(ARTIFACTS),--artifacts $(ARTIFACTS),) \
	      $(if $(ENV),--env $(ENV),) --out $(TABLES)

# Full end-to-end run on a small subset: a few minutes on CPU or a laptop GPU.
# Skips itself when the cached merged dataset is not on this machine.
smoke: ; $(PY) -m pytest -m smoke -q -s

# Unit tests only: about a minute, no data required.
test:  ; $(PY) -m pytest -m "not smoke" -q

# The gene-side foundation models against real checkpoints, on four slides and a few
# hundred spots: minutes on one GPU. `make test` never loads a checkpoint and a
# cluster run is not repeatable while changing code, so this is the rung between.
# Whatever this machine lacks -- an environment, a checkpoint -- is reported and
# skipped, so it is runnable anywhere and says what it could not do.
#
#   make embed-check
#   make embed-check FROM='usz_kidney=/data/TLS_VISIUM_USZ/h5ad_preprocessed/KC*.h5ad'
#   make embed-check CHECK="--strategies global mixed per_slide --repeat"
FROM  ?=
CHECK ?=
# FROM entries are quoted: a value like `usz=/data/KC*.h5ad` is a glob the script
# expands itself, and an unquoted one would be expanded by the shell first, into
# arguments argparse has nowhere to put.
embed-check:
	$(PY) scripts/embed_check.py $(foreach f,$(FROM),--from-h5ad '$(f)') $(CHECK)

# Both halves of the standard: rules, then formatting. `make format` applies it.
lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ; $(PY) -m ruff format .

clean: ; rm -rf artifacts/$(if $(RUN),$(RUN),default)
