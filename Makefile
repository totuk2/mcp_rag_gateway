.PHONY: provision provision-host run up build

# Provision servers/*/manifest.yaml -> registry.generated.yaml + docker-compose.servers.yml
provision:
	python provision.py

# Same, for a gateway running on the host (docker servers publish ports on 127.0.0.1)
provision-host:
	python provision.py --host

# Run the gateway on the host
run:
	python -m gateway

# Bring everything up in docker (gateway + provisioned docker servers on one network)
up: provision
	docker compose -f docker-compose.yml -f docker-compose.servers.yml up --build

build:
	docker compose build
