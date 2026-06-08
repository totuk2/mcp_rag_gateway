.PHONY: provision provision-host run up up-docker build

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

# Same, but provisioning also runs in a container — no host Python/PyYAML needed.
# Step 1 generates registry.generated.yaml + docker-compose.servers.yml; step 2
# builds + runs the gateway and all docker-kind servers together.
up-docker:
	docker compose run --rm provision
	docker compose -f docker-compose.yml -f docker-compose.servers.yml up --build

build:
	docker compose build
