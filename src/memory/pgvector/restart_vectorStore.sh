docker compose down
docker rm pgvector-db
docker volume rm pgvector_pgvector_data
docker compose up -d 
