.PHONY: build up down logs sh test

build:
\tdocker compose build

up:
\tdocker compse up -down

down:
\tdocker compose down

logs:
\tdocker compose logs -f api

sh:
\tdocker compose exec api /bin/bash

test:
\tcurl -s http://localhost:8000/health | jq