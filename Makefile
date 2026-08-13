SERVICES := gateway vision diffusion cpu_tasks
REGISTRY ?= ghcr.io
REPO     ?= your-org/crm-ai-backend
SHA      := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

.PHONY: help test lint dev logs build build-% push-% smoke clean

help:
	@echo "make test              run the gateway test suite"
	@echo "make dev               docker compose up (gateway + cpu)"
	@echo "make build             build every image locally"
	@echo "make build-vision      build one image"
	@echo "make push-vision       build + push one image to the registry"
	@echo "make smoke BASE=... KEY=...   smoke test a live deployment"
	@echo ""
	@echo "  registry: $(REGISTRY)/$(REPO)   sha: $(SHA)"

test:
	pytest tests/ -q

dev:
	docker compose up --build gateway cpu

logs:
	docker compose logs -f --tail=100

build: $(addprefix build-,$(SERVICES))

build-%:
	docker build --platform linux/amd64 \
		-f services/$*/Dockerfile \
		--build-arg BUILD_SHA=$(SHA) \
		-t $(REGISTRY)/$(REPO)/crm-$*:$(SHA) \
		-t crm-$*:local .

push-%: build-%
	docker push $(REGISTRY)/$(REPO)/crm-$*:$(SHA)
	@echo "pushed crm-$*:$(SHA) — point the pod at this tag, never :latest"

smoke:
	@test -n "$(BASE)" || (echo "usage: make smoke BASE=https://... KEY=..."; exit 1)
	python scripts/smoke_test.py --base-url $(BASE) --api-key $(KEY) --concurrency 10

clean:
	docker compose down -v
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
